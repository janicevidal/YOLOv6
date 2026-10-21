# Adapted from Ultralytics TaskAlignedAssigner (AGPL-3.0):
# https://github.com/ultralytics/ultralytics/blob/c7d118b17b30a9f46fc2e11be0334d79acea5af3/ultralytics/utils/tal.py

import math

import torch

from yolov6.assigners.tal_assigner import TaskAlignedAssigner


class STALAssigner(TaskAlignedAssigner):
    """YOLO26 small-target-aware TAL with YOLOv6's four-tuple interface.

    Scores are probabilities; boxes, points and configured FPN strides are
    in pixels. Small GT sides are enlarged to the second FPN stride for
    candidate selection only. Overlaps and targets use the original boxes.
    ``topk2`` filters positives after conflicts; None keeps ``topk``.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0,
                 eps=1e-9, stride=None, topk2=None, use_ciou=True):
        super().__init__(topk, num_classes, alpha, beta, eps)
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError('STAL num_classes must be a positive integer')
        if topk2 is not None and (isinstance(topk2, bool) or
                                  not isinstance(topk2, int) or topk2 <= 0):
            raise ValueError('STAL topk2 must be None or a positive integer')
        if any(not math.isfinite(v) or v < 0 for v in (alpha, beta)):
            raise ValueError('STAL alpha and beta must be nonnegative and finite')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('STAL eps must be positive and finite')
        if not isinstance(use_ciou, bool):
            raise ValueError('STAL use_ciou must be a bool')
        self.stride = tuple(stride) if stride is not None else (8, 16, 32)
        if not self.stride or any(not math.isfinite(v) or v <= 0 for v in self.stride):
            raise ValueError('STAL stride must contain positive finite FPN strides')
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.topk2 = topk if topk2 is None else topk2
        self.use_ciou = use_ciou

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, stride_tensor=None):
        """Return labels [B,N], boxes [B,N,4], scores [B,N,C], mask [B,N].

        Inputs follow TaskAlignedAssigner. ``stride_tensor`` is accepted for
        the shared interface; the minimum GT side uses configured FPN strides.
        """
        # CIoU geometry and the sixth power of small overlaps need FP32 in AMP.
        with torch.cuda.amp.autocast(enabled=False):
            scores, boxes = pd_scores.float(), pd_bboxes.float()
            points, gt_boxes = anc_points.float(), gt_bboxes.float()
            labels = gt_labels.long()
            batch_size, num_anchors = scores.shape[:2]
            self.n_max_boxes = gt_boxes.size(1)
            if batch_size == 0 or num_anchors == 0 or self.n_max_boxes == 0:
                return (torch.full(scores.shape[:2], self.bg_idx, dtype=torch.long,
                                   device=scores.device), torch.zeros_like(boxes),
                        torch.zeros_like(scores), torch.zeros(scores.shape[:2], dtype=torch.bool,
                                                              device=scores.device))

            # Retain YOLOv6 TAL's per-image path for crowded batches.
            chunk_size = 1 if self.n_max_boxes > 100 else batch_size
            results = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                self.bs = end - start
                results.append(self._forward(scores[start:end], boxes[start:end], points,
                                             labels[start:end], gt_boxes[start:end], mask_gt[start:end]))
            return tuple(torch.cat(items, dim=0) for items in zip(*results))

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, align_metric)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize across anchors for EACH GT, as in TAL and Ultralytics.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(-1, keepdim=True).values
        pos_overlaps = (overlaps * mask_pos).max(-1, keepdim=True).values
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2).values
        target_scores = target_scores * norm_align_metric.unsqueeze(-1)
        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        pair_mask = mask_in_gts & mask_gt.bool()
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, pair_mask)
        topk = min(self.topk, pd_scores.size(1))
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, topk).bool())
        return mask_topk * pair_mask, align_metric, overlaps

    def select_candidates_in_gts(self, anc_points, gt_bboxes, mask_gt):
        """Enlarge each short side to stride_val without changing input GTs."""
        centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
        wh = gt_bboxes[..., 2:] - gt_bboxes[..., :2]
        wh = torch.where((wh < self.stride_val) & mask_gt.bool(), wh.new_tensor(self.stride_val), wh)
        lt, rb = centers - wh / 2, centers + wh / 2
        # Keep Ultralytics' strict boundary epsilon independent of norm eps.
        mask = anc_points[:, 0] - lt[..., 0, None] > 1e-9
        mask &= anc_points[:, 1] - lt[..., 1, None] > 1e-9
        mask &= rb[..., 0, None] - anc_points[:, 0] > 1e-9
        mask &= rb[..., 1, None] - anc_points[:, 1] > 1e-9
        return mask

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, pair_mask):
        """Compute metrics only for valid (image, GT, anchor) candidates."""
        batch_idx, gt_idx, anchor_idx = pair_mask.nonzero(as_tuple=True)
        bbox_scores = pd_scores[batch_idx, anchor_idx, gt_labels[batch_idx, gt_idx, 0]]
        overlap_values = self.iou_calculation(gt_bboxes[batch_idx, gt_idx], pd_bboxes[batch_idx, anchor_idx])
        overlaps = pd_bboxes.new_zeros(pair_mask.shape)
        align_metric = pd_scores.new_zeros(pair_mask.shape)
        overlaps[batch_idx, gt_idx, anchor_idx] = overlap_values
        align_metric[batch_idx, gt_idx, anchor_idx] = bbox_scores.pow(self.alpha) * overlap_values.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Ultralytics xyxy bbox_iou, with nonnegative CIoU by default."""
        eps = 1e-7
        x1, y1, x2, y2 = gt_bboxes.chunk(4, -1)
        px1, py1, px2, py2 = pd_bboxes.chunk(4, -1)
        w, h = x2 - x1, y2 - y1 + eps
        pw, ph = px2 - px1, py2 - py1 + eps
        intersection = (x2.minimum(px2) - x1.maximum(px1)).clamp(min=0) * (
            y2.minimum(py2) - y1.maximum(py1)).clamp(min=0)
        iou = intersection / (w * h + pw * ph - intersection + eps)
        if self.use_ciou:
            cw, ch = x2.maximum(px2) - x1.minimum(px1), y2.maximum(py2) - y1.minimum(py1)
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((px1 + px2 - x1 - x2).pow(2) + (py1 + py2 - y1 - y2).pow(2)) / 4
            v = (4 / math.pi ** 2) * ((pw / ph).atan() - (w / h).atan()).pow(2)
            alpha = v / (1 - iou + v + eps)
            iou = iou - (rho2 / c2 + v * alpha)
        return iou.squeeze(-1).clamp(min=0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        topk = min(self.topk, metrics.size(-1))
        topk_metrics, topk_idxs = metrics.topk(topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True).values > self.eps).expand_as(topk_idxs)
        topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)
        # Avoid TAL's [B,G,K,N] one-hot temporary and int8 count overflow.
        counts = torch.zeros_like(metrics, dtype=torch.int32)
        counts.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int32))
        return (counts == 1).to(metrics.dtype)

    def select_highest_overlaps(self, mask_pos, overlaps, align_metric):
        """Resolve conflicts first, then keep each GT's topk2 alignments."""
        multi_gts = (mask_pos.sum(-2, keepdim=True) > 1).expand_as(mask_pos)
        max_overlaps_idx = overlaps.argmax(1, keepdim=True)
        is_max_overlaps = torch.zeros_like(mask_pos).scatter_(1, max_overlaps_idx, 1)
        mask_pos = torch.where(multi_gts, is_max_overlaps, mask_pos)
        if self.topk2 != self.topk:
            metrics = align_metric * mask_pos
            indices = metrics.topk(min(self.topk2, metrics.size(-1)), dim=-1).indices
            mask_pos *= torch.zeros_like(mask_pos).scatter_(-1, indices, 1)
        fg_mask, target_gt_idx = mask_pos.max(-2)
        return target_gt_idx, fg_mask, mask_pos
