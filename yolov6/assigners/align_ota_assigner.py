# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from DAMO-YOLO's damo/base_models/core/ota_assigner.py.
# Batch structure follows MMYOLO's batch_dsl_assigner.py (8c4d9dc503dc).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignOTAAssigner(nn.Module):
    """AlignOTA with MMYOLO-style batch matching and YOLOv6 targets.

    Scores are probabilities; boxes and anchor centers are in image pixels.
    ``stride_tensor`` has shape [num_anchors, 1] (or [num_anchors, 2]).
    Positive class targets are the matched prediction/GT IoUs.
    ``max_cost_elements`` limits each all-class BCE chunk, with one
    anchor/GT/class row as the minimum unit; [B,N,G] metrics stay batched.
    """

    def __init__(self, num_classes=80, center_radius=2.5, candidate_topk=10,
                 iou_weight=3.0, cls_weight=1.0, eps=1e-7,
                 max_cost_elements=4000000):
        super().__init__()
        if (isinstance(candidate_topk, bool) or
                not isinstance(candidate_topk, int) or candidate_topk <= 0):
            raise ValueError('AlignOTA candidate_topk must be a positive integer')
        if not math.isfinite(center_radius) or center_radius <= 0:
            raise ValueError('AlignOTA center_radius must be positive and finite')
        if any(not math.isfinite(v) or v < 0 for v in (iou_weight, cls_weight)):
            raise ValueError('AlignOTA cost weights must be nonnegative and finite')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('AlignOTA eps must be positive and finite')
        if (isinstance(max_cost_elements, bool) or
                not isinstance(max_cost_elements, int) or max_cost_elements <= 0):
            raise ValueError('AlignOTA max_cost_elements must be a positive integer')
        self.num_classes = num_classes
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.eps = eps
        self.max_cost_elements = max_cost_elements

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, stride_tensor=None):
        """Return labels [B,N], boxes [B,N,4], scores [B,N,C], mask [B,N]."""
        if stride_tensor is None:
            raise ValueError('AlignOTA requires per-anchor stride_tensor')
        if stride_tensor.shape not in ((anc_points.size(0), 1),
                                       (anc_points.size(0), 2)):
            raise ValueError('AlignOTA stride_tensor must have shape [N,1] or [N,2]')

        # BCE on probabilities and the finite 1e8 center penalty need FP32 in AMP.
        with torch.cuda.amp.autocast(enabled=False):
            scores, boxes = pd_scores.float(), pd_bboxes.float()
            points, strides = anc_points.float(), stride_tensor.float()
            valid_gt = mask_gt.bool()
            # Padded labels can be -1 and padding need not form a suffix.
            labels = torch.where(valid_gt, gt_labels, torch.zeros_like(gt_labels)).long()
            gt_boxes = gt_bboxes.float().masked_fill(~valid_gt, 0)
            if gt_boxes.size(1) == 0 or boxes.size(0) == 0 or boxes.size(1) == 0:
                return (torch.full(scores.shape[:2], self.num_classes, dtype=torch.long,
                                   device=scores.device), torch.zeros_like(boxes),
                        torch.zeros_like(scores), torch.zeros(scores.shape[:2], dtype=torch.bool,
                                                              device=scores.device))

            self.validate_inputs(scores, boxes, gt_boxes)
            candidates, inside = self.get_candidates(points, strides, gt_boxes, valid_gt)
            cost, overlaps = self.get_box_metrics(scores, boxes, labels, gt_boxes, inside)
            matched_ious, matched_gt, fg_mask = self.dynamic_k_matching(
                cost, overlaps, valid_gt, candidates)
            return self.get_targets(labels, gt_boxes, matched_gt, matched_ious, fg_mask)

    @staticmethod
    def validate_inputs(scores, boxes, gt_boxes):
        """Reject invalid predictions before BCE can poison the CUDA context."""
        tensors = (('pd_scores', scores), ('pd_bboxes', boxes), ('gt_bboxes', gt_boxes))
        valid = ((scores >= 0) & (scores <= 1), torch.isfinite(boxes), torch.isfinite(gt_boxes))
        # One host synchronization for all checks in a healthy batch.
        if torch.stack([mask.all() for mask in valid]).all():
            return
        for (name, tensor), mask in zip(tensors, valid):
            if not mask.all():
                index = tuple((~mask).nonzero(as_tuple=False)[0].tolist())
                value = tensor[index].item()
                requirement = 'finite probabilities in [0, 1]' if name == 'pd_scores' else 'finite coordinates'
                raise FloatingPointError(
                    'AlignOTA requires {}; {}{}={}. Check model outputs, AMP, '
                    'learning rate and upstream losses; this is not a CUDA OOM.'.format(
                        requirement, name, index, value))

    def get_candidates(self, anchor_points, strides, gt_bboxes, mask_gt):
        """Return candidate union [B,N] and box/center intersection [B,N,G]."""
        centers = anchor_points[None, :, None, :]
        box_deltas = torch.cat((centers - gt_bboxes[:, None, :, :2],
                                gt_bboxes[:, None, :, 2:] - centers), dim=-1)
        valid_gt = mask_gt.squeeze(-1)[:, None, :].bool()
        in_boxes = (box_deltas.min(-1).values > 0) & valid_gt
        gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
        radius = self.center_radius * strides[None, :, None, :]
        center_deltas = torch.cat((centers - (gt_centers[:, None] - radius),
                                   gt_centers[:, None] + radius - centers), dim=-1)
        in_centers = (center_deltas.min(-1).values > 0) & valid_gt
        return (in_boxes | in_centers).any(-1), in_boxes & in_centers

    def get_box_metrics(self, scores, boxes, gt_labels, gt_boxes, inside):
        """Build [B,N,G] AlignOTA costs, retaining ALL classification terms."""
        overlaps = self.pairwise_iou(boxes, gt_boxes)
        batch_size, num_anchors, num_classes = scores.shape
        num_gt = gt_boxes.size(1)
        one_hot = F.one_hot(gt_labels.squeeze(-1), num_classes).float()
        flat_scores = scores.reshape(-1, num_classes)
        flat_ious = overlaps.reshape(-1, num_gt)
        cls_cost = torch.empty_like(flat_ious)
        # Unlike DSL's GT-class-only cost, AlignOTA expands all classes.
        # Chunk that temporary tensor instead of allocating [B,N,G,C].
        chunk_size = max(1, self.max_cost_elements // (num_gt * num_classes))
        for start in range(0, batch_size * num_anchors, chunk_size):
            end = min(start + chunk_size, batch_size * num_anchors)
            batch_idx = torch.div(torch.arange(start, end, device=scores.device),
                                  num_anchors, rounding_mode='floor')
            soft_labels = one_hot[batch_idx] * flat_ious[start:end, :, None]
            probabilities = flat_scores[start:end, None, :].expand_as(soft_labels)
            cls_cost[start:end] = (F.binary_cross_entropy(
                probabilities, soft_labels, reduction='none') *
                (soft_labels - probabilities).abs().pow(2)).sum(-1)
        cost = (self.cls_weight * cls_cost.view_as(overlaps) -
                self.iou_weight * torch.log(overlaps + self.eps) +
                (~inside) * 100000000)
        return cost, overlaps

    @staticmethod
    def pairwise_iou(boxes1, boxes2):
        # Match DAMO's union clamp (1e-6); clamp areas for invalid predictions.
        area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0).prod(-1)
        area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0).prod(-1)
        lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
        rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
        intersection = (rb - lt).clamp(min=0).prod(-1)
        union = area1[..., :, None] + area2[..., None, :] - intersection
        return intersection / union.clamp(min=1e-6)

    def dynamic_k_matching(self, cost, pairwise_ious, mask_gt, candidates):
        """Batch sort/dynamic-k, then resolve conflicts across all valid GTs."""
        valid_gt = mask_gt.squeeze(-1).bool()
        pair_valid = candidates[..., None] & valid_gt[:, None, :]
        # Noncandidates must never affect dynamic-k or replace a penalized
        # candidate. Invalid GTs must also be excluded from conflict argmin.
        overlaps = pairwise_ious.masked_fill(~pair_valid, 0)
        cost = cost.masked_fill(~pair_valid, float('inf'))
        counts = candidates.sum(1)
        topk = min(self.candidate_topk, cost.size(1))
        iou_sums = overlaps.topk(topk, dim=1).values.sum(1)
        dynamic_ks = iou_sums.int().clamp(min=1)
        dynamic_ks = torch.minimum(dynamic_ks, counts[:, None]) * valid_gt

        sorted_cost, sorted_indices = torch.sort(cost, dim=1)
        ranks = torch.arange(cost.size(1), device=cost.device)[None, :, None]
        matching = torch.zeros_like(cost, dtype=torch.uint8).scatter_(
            1, sorted_indices, (ranks < dynamic_ks[:, None]).to(torch.uint8))

        # sort and topk can select different members of a boundary tie (often
        # at 1e8). Restore DAMO's original candidate layout/topk only when
        # ambiguous; also preserve rounding for short sets/integer IoU sums.
        boundary = (dynamic_ks.long() - 1).clamp(min=0)
        following = dynamic_ks.long().clamp(max=cost.size(1) - 1)
        ties = (sorted_cost.gather(1, boundary[:, None]) ==
                sorted_cost.gather(1, following[:, None])).squeeze(1)
        ties &= valid_gt & (dynamic_ks > 0) & (dynamic_ks < counts[:, None])
        tolerance = topk * torch.finfo(overlaps.dtype).eps * iou_sums.abs()
        near_integer = ((iou_sums - iou_sums.round()).abs() <= tolerance) & (iou_sums > 1)
        replay = (ties.any(-1) | ((counts > 0) & (counts < self.candidate_topk)) |
                  (near_integer & valid_gt).any(-1))
        self._restore_topk_matches(matching, cost, overlaps, candidates, valid_gt, replay)

        conflicts = matching.sum(2) > 1
        if conflicts.any():
            # As in DAMO, resolve using minimum cost across all GTs.
            min_cost_gt = cost[conflicts].argmin(1)
            matching[conflicts] = 0
            matching[conflicts, min_cost_gt] = 1
        foreground = matching.sum(2) > 0
        matched_gt = matching[foreground].argmax(1)
        matched_iou = (matching * overlaps).sum(2)[foreground]
        return matched_iou, matched_gt, foreground

    def _restore_topk_matches(self, matching, cost, overlaps, candidates, valid_gt, replay):
        """Reproduce per-GT topk for ambiguous images, before batch conflicts."""
        for batch_idx in replay.nonzero(as_tuple=False).flatten().tolist():
            point_indices = candidates[batch_idx].nonzero(as_tuple=False).flatten()
            gt_indices = valid_gt[batch_idx].nonzero(as_tuple=False).flatten()
            single_cost = cost[batch_idx, point_indices][:, gt_indices]
            single_ious = overlaps[batch_idx, point_indices][:, gt_indices]
            topk = min(self.candidate_topk, single_ious.size(0))
            dynamic_ks = single_ious.topk(topk, dim=0).values.sum(0).int().clamp(min=1)
            single_matching = torch.zeros_like(single_cost, dtype=torch.uint8)
            for gt_idx, k in enumerate(dynamic_ks.tolist()):
                pos_idx = single_cost[:, gt_idx].topk(k, largest=False).indices
                single_matching[pos_idx, gt_idx] = 1
            matching[batch_idx] = 0
            matching[batch_idx, point_indices[:, None], gt_indices[None]] = single_matching

    def get_targets(self, gt_labels, gt_bboxes, matched_gt, matched_ious, fg_mask):
        """Gather batch matches into YOLOv6's four target tensors."""
        batch_idx = fg_mask.nonzero(as_tuple=True)[0]
        target_labels = gt_labels.new_full(fg_mask.shape, self.num_classes, dtype=torch.long)
        target_labels[fg_mask] = gt_labels[batch_idx, matched_gt].squeeze(-1)
        target_bboxes = gt_bboxes.new_zeros((*fg_mask.shape, 4))
        target_bboxes[fg_mask] = gt_bboxes[batch_idx, matched_gt]
        target_scores = gt_bboxes.new_zeros((*fg_mask.shape, self.num_classes))
        target_scores[fg_mask] = F.one_hot(target_labels[fg_mask], self.num_classes) * matched_ious[:, None]
        return target_labels, target_bboxes, target_scores, fg_mask
