#!/usr/bin/env python3
"""Person ONNX inference matching detect_online, with official COCO evaluation.

Supports the six activated NCHW heads of yolo_fastest_person0719.onnx. Uses
detect_online's person anchors and integer original-image boxes before NMS.
Defaults: BGR direct resize, confidence .30, NMS IoU .25, up to 10 detections.
RGB and letterbox preprocessing are optional. No PyTorch is needed.
"""

import hashlib
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort

from deploy.ONNX.infer_onnx import ONNXRuntimeSession, build_parser, main, nms
from deploy.ONNX.infer_onnx import preprocess as yolov6_preprocess


# Input-pixel (width, height), ordered by stride 16 then stride 32.
ANCHORS = np.asarray([
    [[9.70, 22.40], [28.50, 63.70], [52.10, 146.70]],
    [[103.41, 238.33], [173.57, 91.55], [246.92, 273.87]],
], dtype=np.float32)
ANCHOR_SOURCE = 'detect_online/source/YXAIHumandtMisc.cpp: bias'


def parse_args(argv=None):
    parser = build_parser(confidence_help='Default: 0.30; use 0.001 for COCO evaluation')
    parser.description = __doc__
    parser.set_defaults(conf_thres=0.30, iou_thres=0.4, max_det=100, output_dir='runs/onnx/fastest')
    parser.add_argument('--resize-mode', choices=('stretch', 'letterbox'), default='stretch',
                        help='Direct resize (default) or aspect-preserving letterbox with padding 114')
    parser.add_argument('--color-order', choices=('bgr', 'rgb'), default='bgr',
                        help='Input channel order (default: bgr, matching detect_online)')
    parser.add_argument('--clip-boxes', action='store_true',
                        help='Clip restored boxes to image boundaries (off by default, matching detect_online)')
    return parser.parse_args(argv)


def preprocess(image, size, dtype, resize_mode='stretch', color_order='bgr'):
    """Direct bilinear resize, BGR CHW, /255; private resize helper is unavailable.

    detect_online passes CV_BGR to uint8RGBToCHWFloatRGB: it preserves the BGR
    bytes, despite the helper's RGB name, and multiplies by a float32 1/255.
    """
    if resize_mode == 'letterbox':
        # Reuse exactly the existing demo's dimensions, rounding and fill 114.
        if color_order == 'bgr':
            image = image[:, :, ::-1]  # existing helper reverses channels again
        return yolov6_preprocess(image, size, dtype)
    height, width = image.shape[:2]
    resized = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    if color_order == 'rgb':
        resized = resized[:, :, ::-1]
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=dtype)
    tensor *= np.float32(1 / 255)
    return tensor, (size[1] / width, size[0] / height, 0, 0)


def restore_boxes(rows, image_shape, transform, iou_thres, max_det, clip=False):
    """C++ generateBBox -> nmsBBox -> top10, with configurable evaluation limits.

    Input rows contain cx,cy,w,h,score,class. BBox's int32 corners truncate toward
    zero *after* scaling to the original image; NMS therefore uses integer boxes.
    The default preserves out-of-image coordinates, including negative corners.
    """
    gain_x, gain_y, left, top = transform
    scale = np.asarray([1 / gain_x, 1 / gain_y], dtype=np.float32)
    xywh = rows[:, :4].astype(np.float64)
    low = (xywh[:, :2] - 0.5 * xywh[:, 2:4] - (left, top)) * scale
    high = (xywh[:, :2] + 0.5 * xywh[:, 2:4] - (left, top)) * scale
    rows[:, :4] = np.trunc(np.concatenate((low, high), axis=1)).astype(np.int32)
    if clip:
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, image_shape[1])
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, image_shape[0])
    return nms(rows, iou_thres, max_det)


def decode_outputs(outputs, size):
    """Decode activated NCHW heads to anchor-major [cx,cy,w,h,score,class_id]."""
    predictions = []
    for level in range(2):
        reg, obj, cls = [np.asarray(value, dtype=np.float32)
                         for value in outputs[level * 3:level * 3 + 3]]
        batch, channels, height, width = reg.shape
        if (height * (16 * 2 ** level), width * (16 * 2 ** level)) != tuple(size):
            raise ValueError('Expected heads ordered by strides 16 and 32')
        if channels != 12 or obj.shape != (batch, 3, height, width) or cls.shape != (batch, 1, height, width):
            raise ValueError('Expected NCHW reg=12, obj=3 and one shared per-cell person class')
        if any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
               for value in (reg, obj, cls)) or not np.allclose(cls, 1, atol=1e-4):
            raise ValueError('Expected activated sigmoid reg/obj and softmax cls outputs; logits are unsupported')
        reg = reg.reshape(batch, 3, 4, height, width).transpose(0, 1, 3, 4, 2)
        gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        grid = np.stack((gx, gy), axis=-1).astype(np.float32)[None, None, :, :, :]
        # C++ uses double literals/pow, then stores cx/cy/w/h as float. Preserve
        # that order to avoid crossing integer boundaries during restoration.
        geometry = reg.astype(np.float64)
        xy = (geometry[..., :2] * 2 - 0.5 + grid) * (size[0] / height)
        wh = np.square(geometry[..., 2:4] * 2) * ANCHORS[level, :, None, None, :]
        values = np.empty((batch, 3, height, width, 6), dtype=np.float32)
        values[..., :2], values[..., 2:4] = xy, wh
        values[..., 4], values[..., 5] = obj * cls, 0
        predictions.append(values.reshape(batch, -1, values.shape[-1]))
    return np.concatenate(predictions, axis=1)


class FastestONNXDetector(ONNXRuntimeSession):
    def __init__(self, args):
        super().__init__(args)
        if self.size[0] % 32 or self.size[1] % 32:
            raise ValueError('Input height/width must be multiples of 32')
        if self.size[0] != self.size[1]:
            raise ValueError('detect_online indexes its grid assuming square heads; use a square input')
        canonical = [f'out_{kind}_{level}' for level in (2, 3) for kind in ('reg', 'obj', 'cls')]
        if len(self.outputs) == 6 and all(len(item.shape) == 4 for item in self.outputs):
            if set(canonical) == set(self.output_names):
                self.output_names = canonical
            meta = {item.name: item for item in self.outputs}
            counts = [meta[self.output_names[index]].shape[1] for index in (2, 5)]
        else:
            raise ValueError('Expected six activated NCHW outputs: reg/obj/cls for strides 16 and 32')
        if counts != [1, 1] or args.num_classes not in (None, 1):
            raise ValueError('This demo supports the single person class only')
        self.num_classes = 1
        self.format = 'fastestv2-split'
        self.extra_summary = {
            'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            'onnxruntime_version': ort.__version__,
            'threads': args.threads,
            'anchors': ANCHORS.tolist(), 'anchor_source': ANCHOR_SOURCE,
            'pipeline': 'detect-online',
            'resize_mode': args.resize_mode, 'color_order': args.color_order,
            'clip_boxes': args.clip_boxes, 'num_classes': self.num_classes,
            'decoder': 'xy=(2*reg_xy-0.5+grid)*stride; wh=(2*reg_wh)^2*anchors',
            'score': 'objectness*person_class_probability',
            'box_coordinates': 'truncate original-image corners toward zero to int32 before NMS',
            'resize_implementation': 'OpenCV INTER_LINEAR (detect_online private bilinear helper unavailable)',
            'normalization': ('division by 255' if args.resize_mode == 'letterbox'
                              else 'float32 multiplication by 1/255'),
            'nms_stage': 'original-image restoration',
        }
        print(f'ONNX input={self.input.shape}, format={self.format}, classes={self.num_classes}, '
              f'preprocess={args.resize_mode}/{args.color_order}, '
              f'providers={self.session.get_providers()}')

    def predict(self, tensors):
        outputs = self.session.run(self.output_names, {self.input.name: np.stack(tensors)})
        predictions = decode_outputs(outputs, self.size)
        # Integer NMS runs in restore_boxes when the original image size is known.
        return [rows[rows[:, 4] > self.args.conf_thres] for rows in predictions]


if __name__ == '__main__':
    args = parse_args()
    main(args, detector_class=FastestONNXDetector,
         preprocess_fn=partial(preprocess, resize_mode=args.resize_mode,
                               color_order=args.color_order),
         restore_fn=partial(restore_boxes, iou_thres=args.iou_thres,
                            max_det=args.max_det, clip=args.clip_boxes))
