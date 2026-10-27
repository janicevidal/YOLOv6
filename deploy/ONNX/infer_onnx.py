#!/usr/bin/env python3
"""Folder inference and official COCO bbox evaluation with ONNX Runtime.

Runtime dependencies: numpy, opencv-python, onnxruntime; pycocotools for evaluation.
No PyTorch installation or YOLOv6 imports are needed.
"""

import argparse
import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
COCO_METRICS = (
    'mAP_0.50_0.95', 'mAP_0.50', 'mAP_0.75', 'AP_small', 'AP_medium', 'AP_large',
    'AR_maxDets_1', 'AR_maxDets_10', 'AR_maxDets_100',
    'AR_small', 'AR_medium', 'AR_large',
)


def build_parser(confidence_help='Default: 0.001 with annotations, otherwise 0.25'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='ONNX model file')
    parser.add_argument('--source', required=True, help='Image directory (searched recursively)')
    parser.add_argument('--output-dir', default='runs/onnx/exp')
    parser.add_argument('--annotations', help='COCO instances JSON; enables official bbox evaluation')
    parser.add_argument('--category-ids', nargs='+', type=int,
                        help='COCO category IDs in model class order, e.g. --category-ids 1')
    parser.add_argument('--names', nargs='+', help='Display names in model class order')
    parser.add_argument('--num-classes', type=int,
                        help='Class count; required for end-to-end outputs, otherwise an optional check')
    parser.add_argument('--img-size', nargs='+', type=int,
                        help='HEIGHT WIDTH (or one square size); inferred for static ONNX inputs')
    parser.add_argument('--batch-size', type=int,
                        help='Dynamic ONNX batch size; static ONNX must use its exported batch size')
    parser.add_argument('--device', default='cpu', help='cpu or CUDA device index, e.g. 0')
    parser.add_argument('--threads', type=int, default=4, help='ORT CPU intra-op threads')
    parser.add_argument('--conf-thres', type=float, help=confidence_help)
    parser.add_argument('--iou-thres', type=float, default=0.65, help='Class-aware NMS IoU')
    parser.add_argument('--max-det', type=int, default=100, help='Detections per image after NMS')
    parser.add_argument('--vis-conf', type=float, default=0.25,
                        help='Drawing threshold only; does not filter predictions.json or COCO eval')
    parser.add_argument('--no-save-images', action='store_true', help='Skip annotated image output')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save class_id score x1 y1 x2 y2 in original-image pixels')
    return parser


def parse_args():
    parser = build_parser()
    parser.add_argument('--resize-mode', choices=('letterbox', 'stretch'), default='letterbox',
                        help='Aspect-preserving letterbox (default) or direct resize without padding')
    return parser.parse_args()


def collect_images(source, output_dir, annotations=None):
    """Use every annotated image, including images that will have no detections."""
    if not source.is_dir():
        raise ValueError(f'Image directory does not exist: {source}')
    paths = sorted(p for p in source.rglob('*') if p.is_file()
                   and p.suffix.lower() in IMAGE_SUFFIXES
                   and output_dir not in p.parents)
    if not paths:
        raise ValueError(f'No images found in {source}')
    if annotations is None:
        return [(p, p.relative_to(source).as_posix()) for p in paths]
    relative = {p.relative_to(source).as_posix(): p for p in paths}
    basenames = defaultdict(list)
    for p in paths:
        basenames[p.name].append(p)
    records, missing = [], []
    seen_ids, seen_paths = set(), set()
    for item in annotations['images']:
        name = item['file_name'].replace('\\', '/')
        path = relative.get(name)
        if path is None:
            matches = basenames.get(Path(name).name, [])
            if len(matches) > 1:
                raise ValueError(f'Ambiguous COCO file_name: {name}; preserve annotation subdirectories')
            path = matches[0] if matches else None
        if path is None:
            missing.append(name)
            continue
        if item['id'] in seen_ids or path in seen_paths:
            raise ValueError(f'Duplicate image ID or image path in COCO annotations: {name}')
        seen_ids.add(item['id'])
        seen_paths.add(path)
        records.append((path, item['id']))
    if missing:
        raise ValueError(f'{len(missing)} COCO images are missing from --source, e.g. {missing[:3]}. '
                         'Provide the complete validation folder or an explicit subset annotation JSON.')
    if not records:
        raise ValueError('COCO annotation JSON contains no images')
    return records


def preprocess(image, size, dtype, resize_mode='letterbox'):
    """Resize, BGR -> RGB, CHW, /255; return (gain_x, gain_y, left, top)."""
    height, width = image.shape[:2]
    if resize_mode == 'stretch':
        image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        transform = (size[1] / width, size[0] / height, 0, 0)
    elif resize_mode == 'letterbox':
        ratio = min(size[0] / height, size[1] / width)
        resized = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        if resized != (width, height):
            image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
        dw, dh = (size[1] - resized[0]) / 2, (size[0] - resized[1]) / 2
        left, top = round(dw - 0.1), round(dh - 0.1)
        right, bottom = round(dw + 0.1), round(dh + 0.1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                  value=(114, 114, 114))
        transform = (ratio, ratio, left, top)
    else:
        raise ValueError(f'Unsupported resize mode: {resize_mode}')
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=dtype)
    tensor /= 255
    return tensor, transform


def nms(detections, iou_thres, max_det):
    """Class-aware NMS for [x1,y1,x2,y2,score,class_id], without torchvision."""
    if not len(detections):
        return np.empty((0, 6), dtype=np.float32)
    detections = detections[np.argsort(-detections[:, 4], kind='stable')[:30000]]
    keep = []
    boxes = detections[:, :4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for class_id in np.unique(detections[:, 5]):
        order = np.flatnonzero(detections[:, 5] == class_id)
        class_kept = 0
        while order.size and class_kept < max_det:
            current = order[0]
            keep.append(current)
            class_kept += 1
            if order.size == 1:
                break
            rest = order[1:]
            wh = np.maximum(0, np.minimum(boxes[current, 2:], boxes[rest, 2:])
                            - np.maximum(boxes[current, :2], boxes[rest, :2]))
            intersection = wh[:, 0] * wh[:, 1]
            iou = intersection / np.maximum(areas[current] + areas[rest] - intersection, 1e-7)
            order = rest[iou <= iou_thres]
    keep = np.asarray(keep)
    keep = keep[np.argsort(-detections[keep, 4], kind='stable')[:max_det]]
    return detections[keep]


def filter_predictions(boxes, scores, conf_thres, iou_thres, max_det):
    # Multi-label filtering matches the project's validation NMS.
    box_ids, class_ids = np.nonzero(scores > conf_thres)
    detections = np.column_stack((boxes[box_ids], scores[box_ids, class_ids], class_ids))
    valid = np.isfinite(detections).all(axis=1)
    valid &= (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
    return nms(detections[valid].astype(np.float32), iou_thres, max_det)


class ONNXRuntimeSession:
    """Shared session/input setup for the YOLOv6 and FastestV2 demos."""
    def __init__(self, args):
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        if args.device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            device_id = int(args.device)
            if device_id < 0 or 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise ValueError('CUDA is unavailable; install compatible onnxruntime-gpu or use --device cpu')
            providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        self.session = ort.InferenceSession(args.model, sess_options=options, providers=providers)
        if args.device != 'cpu' and 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('ORT fell back to CPU; check the onnxruntime-gpu/CUDA/cuDNN installation')
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError('Expected one NCHW image input')
        self.input = inputs[0]
        shape = self.input.shape
        if isinstance(shape[1], int) and shape[1] != 3:
            raise ValueError(f'Expected a three-channel input: {shape}')
        types = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}
        if self.input.type not in types:
            raise ValueError('Expected FP32/FP16 input; models with embedded preprocessing are unsupported')
        self.dtype = types[self.input.type]
        size = args.img_size
        if size is not None:
            if len(size) not in (1, 2) or min(size) <= 0:
                raise ValueError('--img-size must be one positive size or HEIGHT WIDTH')
            size = tuple(size * 2 if len(size) == 1 else size)
        elif all(isinstance(v, int) and v > 0 for v in shape[2:]):
            size = tuple(shape[2:])
        else:
            raise ValueError('Dynamic spatial dimensions require --img-size HEIGHT WIDTH')
        if any(isinstance(v, int) and v > 0 and v != s for v, s in zip(shape[2:], size)):
            raise ValueError(f'--img-size {size} differs from static ONNX input {shape}')
        self.size = size
        self.static_batch = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else None
        if self.static_batch and args.batch_size not in (None, self.static_batch):
            raise ValueError(f'ONNX batch is fixed at {self.static_batch}; omit --batch-size or export '
                             'again with --dynamic-batch. Folder processing works with batch=1.')
        self.batch_size = args.batch_size or self.static_batch or 1
        self.outputs = self.session.get_outputs()
        self.output_names = [output.name for output in self.outputs]
        self.args = args


class ONNXDetector(ONNXRuntimeSession):
    def __init__(self, args):
        super().__init__(args)
        outputs = self.outputs
        if {'num_dets', 'det_boxes', 'det_scores', 'det_classes'} <= set(self.output_names):
            self.format = 'end2end'
            self.num_classes = args.num_classes or len(args.category_ids or args.names or [])
            if not self.num_classes:
                raise ValueError('End-to-end output needs --num-classes, --names or --category-ids')
        elif len(outputs) == 3 and all(len(output.shape) == 4 for output in outputs):
            self.format = 'raw'
            self.num_classes = None
        elif len(outputs[0].shape) == 3:
            self.format = 'decoded'
            self.num_classes = None
        else:
            raise ValueError(f'Unsupported ONNX outputs: {[(x.name, x.shape) for x in outputs]}')
        print(f'ONNX: input={self.input.shape}, format={self.format}, size={self.size}, '
              f'batch={self.batch_size}, providers={self.session.get_providers()}')
        if self.format == 'end2end':
            print('End-to-end ONNX already includes NMS; its exported thresholds/topk remain in effect.')

    def predict(self, tensors):
        outputs = self.session.run(None, {self.input.name: np.stack(tensors)})
        args = self.args
        if self.format == 'end2end':
            values = dict(zip(self.output_names, outputs))
            detections = []
            for i, count in enumerate(values['num_dets'].reshape(-1)):
                count = int(count)
                rows = np.column_stack((values['det_boxes'][i, :count],
                                        values['det_scores'][i, :count].reshape(-1),
                                        values['det_classes'][i, :count].reshape(-1))).astype(np.float32)
                rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] > args.conf_thres)]
                rows = rows[np.argsort(-rows[:, 4], kind='stable')[:args.max_det]]
                detections.append(rows)
            return detections
        if self.format == 'raw':
            all_boxes, all_scores = [], []
            class_counts = set()
            for output in outputs:
                output = output.astype(np.float32)
                batch, channels, height, width = output.shape
                class_counts.add(channels - 4)
                sy, sx = self.size[0] / height, self.size[1] / width
                if abs(sy - sx) > 1e-6:
                    raise ValueError('Raw head feature dimensions do not match the input strides')
                grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                centers = np.stack((grid_x, grid_y), axis=-1).reshape(1, -1, 2).astype(np.float32) + 0.5
                values = output.transpose(0, 2, 3, 1).reshape(batch, -1, channels)
                # Pico exports probabilities followed by raw LTRB, with no extra sigmoid.
                distances = values[..., -4:]
                boxes = np.concatenate((centers - distances[..., :2],
                                        centers + distances[..., 2:]), axis=-1) * sx
                all_boxes.append(boxes)
                all_scores.append(values[..., :-4])
            if len(class_counts) != 1 or min(class_counts) < 1:
                raise ValueError('Raw heads have inconsistent class counts')
            self.num_classes = class_counts.pop()
            boxes = np.concatenate(all_boxes, axis=1)
            scores = np.concatenate(all_scores, axis=1)
        else:
            prediction = outputs[0].astype(np.float32)
            self.num_classes = prediction.shape[-1] - 5
            if self.num_classes < 1:
                raise ValueError('Decoded output must be [B,N,5+num_classes]')
            boxes = np.concatenate((prediction[..., :2] - prediction[..., 2:4] / 2,
                                    prediction[..., :2] + prediction[..., 2:4] / 2), axis=-1)
            scores = prediction[..., 5:] * prediction[..., 4:5]
        if args.num_classes is not None and args.num_classes != self.num_classes:
            raise ValueError('--num-classes does not match ONNX outputs')
        return [filter_predictions(b, s, args.conf_thres, args.iou_thres, args.max_det)
                for b, s in zip(boxes, scores)]


def category_mapping(num_classes, args, annotations):
    categories = {item['id']: item['name'] for item in annotations['categories']} if annotations else {}
    if args.category_ids is not None:
        ids = args.category_ids
    elif annotations:
        if len(categories) != num_classes:
            raise ValueError(f'Model has {num_classes} classes but annotations have {len(categories)} categories. '
                             'Set --category-ids in model order (person on full COCO: --category-ids 1).')
        ids = sorted(categories)
    else:
        ids = list(range(num_classes))
    if len(ids) != num_classes or len(set(ids)) != len(ids):
        raise ValueError('--category-ids must contain one unique ID per model class')
    if annotations and any(i not in categories for i in ids):
        raise ValueError('--category-ids contains an ID absent from the COCO categories')
    names = args.names or [categories.get(i, f'class_{index}') for index, i in enumerate(ids)]
    if len(names) != num_classes:
        raise ValueError('--names must contain one name per model class')
    print(f'Class mapping: {[(i, ids[i], names[i]) for i in range(num_classes)]}')
    return ids, names


def save_image(image, detections, names, path, threshold):
    for x1, y1, x2, y2, score, class_id in detections:
        if score < threshold:
            continue
        class_id = int(class_id)
        color = ((37 * class_id + 60) % 256, (17 * class_id + 190) % 256, (29 * class_id + 80) % 256)
        a, b = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(image, a, b, color, 2)
        cv2.putText(image, f'{names[class_id]} {score:.2f}', (a[0], max(15, a[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write image: {path}')


def evaluate_coco(annotation_path, predictions, image_ids, category_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(annotation_path))
    if predictions:
        result = ground_truth.loadRes(predictions)
    else:
        # COCO.loadRes([]) indexes the first detection in several released versions.
        result = COCO()
        result.dataset = {'images': ground_truth.dataset['images'],
                          'categories': ground_truth.dataset['categories'], 'annotations': []}
        result.createIndex()
    evaluator = COCOeval(ground_truth, result, 'bbox')
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    # Keep official IoUs 0.50:0.05:0.95, area ranges and maxDets=[1,10,100].
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(zip(COCO_METRICS, map(float, evaluator.stats)))


def restore_boxes(rows, image_shape, transform):
    gain_x, gain_y, left, top = transform
    rows[:, [0, 2]] = ((rows[:, [0, 2]] - left) / gain_x).clip(0, image_shape[1])
    rows[:, [1, 3]] = ((rows[:, [1, 3]] - top) / gain_y).clip(0, image_shape[0])
    return rows


def main(args, detector_class=ONNXDetector, preprocess_fn=None, restore_fn=restore_boxes):
    if preprocess_fn is None:
        preprocess_fn = partial(preprocess, resize_mode=getattr(args, 'resize_mode', 'letterbox'))
    if args.conf_thres is None:
        args.conf_thres = 0.001 if args.annotations else 0.25
    if not all(0 <= value <= 1 for value in (args.conf_thres, args.iou_thres, args.vis_conf)):
        raise ValueError('Confidence and IoU thresholds must be in [0,1]')
    if args.max_det < 1 or args.threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError('--max-det, --threads and --batch-size must be positive')
    if args.num_classes is not None and args.num_classes < 1:
        raise ValueError('--num-classes must be positive')
    cv2.setNumThreads(1)
    source, output_dir = Path(args.source).resolve(), Path(args.output_dir).resolve()
    if source == output_dir:
        raise ValueError('--output-dir must differ from --source')
    annotations = json.loads(Path(args.annotations).read_text()) if args.annotations else None
    if annotations:
        try:
            import pycocotools.cocoeval  # noqa: F401
        except ImportError as error:
            raise ImportError('COCO evaluation requires: pip install pycocotools') from error
    records = collect_images(source, output_dir, annotations)
    detector = detector_class(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.annotations and args.conf_thres > 0.01:
        print('Evaluation confidence threshold is high; use --conf-thres 0.001 to preserve recall.')
    predictions, detections_json = [], []
    category_ids, names = None, None
    inference_seconds = 0.0
    restoration_seconds = 0.0
    started = time.perf_counter()
    for start in range(0, len(records), detector.batch_size):
        batch = records[start:start + detector.batch_size]
        images, tensors, transforms = [], [], []
        for path, _ in batch:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Could not read image: {path}')
            tensor, transform = preprocess_fn(image, detector.size, detector.dtype)
            images.append(image)
            tensors.append(tensor)
            transforms.append(transform)
        valid_count = len(tensors)
        if detector.static_batch:
            tensors += [tensors[-1]] * (detector.static_batch - valid_count)
        tick = time.perf_counter()
        outputs = detector.predict(tensors)
        inference_seconds += time.perf_counter() - tick
        if category_ids is None:
            category_ids, names = category_mapping(detector.num_classes, args, annotations)
        for (path, image_id), image, transform, rows in zip(batch, images, transforms, outputs[:valid_count]):
            if len(rows) and (np.any(rows[:, 5] < 0) or np.any(rows[:, 5] >= len(category_ids))):
                raise ValueError('Predicted class ID is outside the configured class mapping')
            tick = time.perf_counter()
            rows = restore_fn(rows, image.shape[:2], transform)
            restoration_seconds += time.perf_counter() - tick
            rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
            relative = path.relative_to(source)
            per_image = []
            for x1, y1, x2, y2, score, class_id in rows:
                class_id = int(class_id)
                box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                predictions.append({'image_id': image_id, 'category_id': category_ids[class_id],
                                    'bbox': box, 'score': float(score)})
                per_image.append({'class_id': class_id, 'category_id': category_ids[class_id],
                                  'name': names[class_id], 'score': float(score),
                                  'bbox_xyxy': list(map(float, (x1, y1, x2, y2)))})
            detections_json.append({'file_name': relative.as_posix(), 'image_id': image_id,
                                    'width': image.shape[1], 'height': image.shape[0],
                                    'detections': per_image})
            if args.save_txt:
                target = (output_dir / 'labels' / relative).with_suffix(relative.suffix + '.txt')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(''.join(f'{int(row[5])} {row[4]:.7f} '
                                          + ' '.join(f'{v:.4f}' for v in row[:4]) + '\n' for row in rows))
            if not args.no_save_images:
                save_image(image, rows, names, output_dir / 'images' / relative, args.vis_conf)
        done = start + valid_count
        if start == 0 or done == len(records) or done // 100 != start // 100:
            print(f'Processed {done}/{len(records)} images, detections={len(predictions)}', flush=True)
    (output_dir / 'predictions.json').write_text(json.dumps(predictions))
    (output_dir / 'detections.json').write_text(json.dumps(detections_json, ensure_ascii=False))
    summary = {'model': str(Path(args.model).resolve()), 'source': str(source),
               'annotations': str(Path(args.annotations).resolve()) if args.annotations else None,
               'format': detector.format, 'image_count': len(records), 'detection_count': len(predictions),
               'input_size': detector.size, 'batch_size': detector.batch_size,
               'resize_mode': getattr(args, 'resize_mode', 'letterbox'),
               'providers': detector.session.get_providers(),
               'category_ids': category_ids, 'names': names,
               'conf_thres': args.conf_thres, 'iou_thres': args.iou_thres, 'max_det': args.max_det,
               'inference_and_nms_seconds': inference_seconds + restoration_seconds,
               'restoration_and_finalization_seconds': restoration_seconds,
               'total_inference_pipeline_seconds': time.perf_counter() - started}
    summary.update(getattr(detector, 'extra_summary', {}))
    if args.annotations:
        summary['metrics'] = evaluate_coco(args.annotations, predictions,
                                           [image_id for _, image_id in records], category_ids)
        print(f"mAP@0.5: {summary['metrics']['mAP_0.50']:.6f} | "
              f"mAP@0.50:0.95: {summary['metrics']['mAP_0.50_0.95']:.6f}")
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main(parse_args())
