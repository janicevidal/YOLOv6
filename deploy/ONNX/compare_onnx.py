#!/usr/bin/env python3
"""Compare ONNX Runtime and checkpoint COCO AP using YOLOv6 validation code.

Square mode evaluates the supplied ONNX. --rect --export-dynamic creates a
separate Pico ONNX for the variable image shapes used during training validation.
Requires the project/PyTorch dependencies, onnxruntime and pycocotools.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from tqdm import tqdm

from deploy.ONNX.infer_onnx import category_mapping, collect_images, evaluate_coco
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.core.evaler import Evaler
from yolov6.data.datasets import TrainValDataset
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.config import Config
from yolov6.utils.general import dist2bbox
from yolov6.utils.nms import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Original ONNX, without embedded NMS/preprocessing')
    parser.add_argument('--weights', help='Matching .pt; defaults to the ONNX path with .pt suffix')
    parser.add_argument('--data', required=True, help='Dataset YAML with nc, names, val and anno_path')
    parser.add_argument('--source', help='Override val image directory in YAML')
    parser.add_argument('--annotations', help='Override COCO anno_path in YAML')
    parser.add_argument('--category-ids', nargs='+', type=int, help='Annotation IDs in model class order')
    parser.add_argument('--img-size', type=int, help='Square size; inferred from static ONNX or defaults to 416')
    parser.add_argument('--rect', action='store_true', help='Training validation: rect=True, pad=0.5')
    parser.add_argument('--export-dynamic', action='store_true',
                        help='Export a separate dynamic B/H/W Pico ONNX from --weights, then compare it')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Validation grouping size; use training batch_size/world_size*2 for --rect')
    parser.add_argument('--infer-batch-size', type=int, default=1,
                        help='Inference microbatch shared by both backends; independent of rect grouping')
    parser.add_argument('--config-file', help='Optional training config: reuse data_aug in the val loader')
    parser.add_argument('--conf-thres', type=float, default=0.03, help='Training validation default: 0.03')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='Training validation default: 0.65')
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--device', default='cpu', help='Device for both backends and NMS: cpu or CUDA index')
    parser.add_argument('--threads', type=int, default=2, help='CPU threads per backend')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3, help='Unmeasured runs on the first input shape')
    parser.add_argument('--limit', type=int, help='Debug: evaluate only the first N loader images')
    parser.add_argument('--output-dir', default='runs/onnx/compare')
    return parser.parse_args()


def file_info(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def export_dynamic(weights, destination, img_size):
    """Use the real exporter conversions; never alter dimensions of a static graph."""
    from deploy.ONNX.export_onnx import prepare_model_for_export, select_opset

    model = prepare_model_for_export(load_checkpoint(str(weights), map_location='cpu'), raw_head=True)
    axes = {'images': {0: 'batch', 2: 'height', 3: 'width'}}
    for name in ('p3', 'p4', 'p5'):
        axes[name] = {0: 'batch', 2: name + '_height', 3: name + '_width'}
    torch.onnx.export(model, torch.zeros(1, 3, img_size, img_size), str(destination),
                      opset_version=select_opset(model), input_names=['images'],
                      output_names=['p3', 'p4', 'p5'], dynamic_axes=axes,
                      do_constant_folding=True)
    import onnx

    onnx.checker.check_model(onnx.load(str(destination)))


class ORTBackend:
    def __init__(self, path, device, threads, num_classes, strides):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = ['CPUExecutionProvider']
        if device.type == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDAExecutionProvider unavailable; install compatible onnxruntime-gpu')
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': device.index}))
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device.type == 'cuda' and 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('ORT fell back to CPU; use --device cpu or fix CUDA/cuDNN dependencies')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].type != 'tensor(float)':
            raise ValueError('Comparison requires one FP32 NCHW input; use an FP32 export')
        self.input = inputs[0]
        self.output_names = [item.name for item in outputs]
        self.raw = len(outputs) == 3 and all(len(item.shape) == 4 for item in outputs)
        if not self.raw and not (len(outputs) == 1 and len(outputs[0].shape) == 3):
            raise ValueError('Expected raw Pico heads or decoded [B,N,5+nc]; embedded NMS is unsupported')
        self.device, self.nc, self.strides = device, num_classes, strides
        self.static_batch = self.input.shape[0] if isinstance(self.input.shape[0], int) else None

    def __call__(self, images):
        shape = self.input.shape
        for actual, expected in zip(images.shape[1:], shape[1:]):
            if isinstance(expected, int) and actual != expected:
                raise ValueError(f'Input {tuple(images.shape)} incompatible with ONNX {shape}; '
                                 'use --export-dynamic for --rect')
        count = len(images)
        if self.static_batch:
            if count > self.static_batch:
                raise ValueError('--infer-batch-size exceeds static ONNX batch')
            if count < self.static_batch:
                images = torch.cat((images, images[-1:].expand(self.static_batch - count, -1, -1, -1)))
        arrays = self.session.run(self.output_names, {self.input.name: images.cpu().numpy()})
        outputs = [torch.from_numpy(value[:count]).to(self.device) for value in arrays]
        if not self.raw:
            if outputs[0].shape[-1] != self.nc + 5:
                raise ValueError('Decoded ONNX class count differs from dataset')
            return outputs[0]
        for output, stride in zip(outputs, self.strides):
            if output.shape[1] != self.nc + 4 or tuple(output.shape[2:]) != (
                    images.shape[2] // stride, images.shape[3] // stride):
                raise ValueError('Expected raw P3/P4/P5 ordered by checkpoint strides, channels=[scores,LTRB]')
        # Use the same anchor generation and LTRB -> xywh as Pico Detect.forward.
        # ONNX scores already contain sigmoid. Distances are in grid units.
        values = torch.cat([value.flatten(2).permute(0, 2, 1) for value in outputs], dim=1)
        points, stride_tensor = generate_anchors(outputs, self.strides, device=self.device, is_eval=True)
        boxes = dist2bbox(values[..., self.nc:], points, box_format='xywh') * stride_tensor
        return torch.cat((boxes, torch.ones_like(boxes[..., :1]), values[..., :self.nc]), dim=-1)


class PredictionDifference:
    """Compare corresponding anchors before confidence filtering/NMS."""
    def __init__(self):
        self.total = {'box': 0.0, 'score': 0.0}
        self.maximum = {'box': 0.0, 'score': 0.0}
        self.count = {'box': 0, 'score': 0}

    def update(self, native, onnx):
        if native.shape != onnx.shape:
            raise ValueError(f'Prediction shapes differ: PyTorch {native.shape}, ONNX {onnx.shape}')
        if not torch.isfinite(native).all() or not torch.isfinite(onnx).all():
            raise ValueError('Non-finite predictions encountered')
        for name, columns in (('box', slice(0, 4)), ('score', slice(5, None))):
            diff = (native[..., columns] - onnx[..., columns]).abs()
            self.total[name] += diff.sum().item()
            self.maximum[name] = max(self.maximum[name], diff.max().item())
            self.count[name] += diff.numel()

    def result(self):
        return {name: {'mean_abs': self.total[name] / self.count[name], 'max_abs': self.maximum[name]}
                for name in self.total}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def compare(args):
    if min(args.batch_size, args.infer_batch_size, args.threads, args.max_det) <= 0:
        raise ValueError('Batch sizes, threads and max-det must be positive')
    if args.workers < 0 or args.warmup < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError('workers/warmup must be nonnegative; limit must be positive')
    if not all(0 <= v <= 1 for v in (args.conf_thres, args.iou_thres)):
        raise ValueError('Confidence and IoU thresholds must be within [0,1]')
    torch.set_num_threads(args.threads)
    device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{int(args.device)}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    original = Path(args.model)
    weights = Path(args.weights) if args.weights else original.with_suffix('.pt')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.data) as stream:
        data = yaml.safe_load(stream)
    source = Path(args.source or data['val']).resolve()
    annotation_path = Path(args.annotations or data['anno_path']).resolve()
    annotations = json.loads(annotation_path.read_text())
    records = collect_images(source, output_dir, annotations)
    image_ids = {str(path.resolve()): image_id for path, image_id in records}
    args.names = data['names']
    category_ids, _ = category_mapping(data['nc'], args, annotations)

    model = load_checkpoint(str(weights), map_location='cpu', fuse=False).float().eval()
    # Match training validation, retaining unfused Pico/Conv/BN branches.
    model.export = False
    model.detect.export = False
    if model.detect.nc != data['nc']:
        raise ValueError('Checkpoint class count differs from dataset')
    strides = [int(value) for value in model.detect.stride]
    model = model.to(device)
    original_info = ORTBackend(original, device, args.threads, data['nc'], strides)
    static_hw = original_info.input.shape[2:]
    default_size = static_hw[0] if isinstance(static_hw[0], int) else 416
    img_size = args.img_size or default_size
    if img_size <= 0 or img_size % max(strides):
        raise ValueError('--img-size must be a positive multiple of the largest stride')
    active_path, backend = original, original_info
    if args.export_dynamic:
        active_path = output_dir / 'dynamic_raw.onnx'
        if active_path.resolve() == original.resolve():
            raise ValueError('The generated ONNX must be separate from --model')
        export_dynamic(weights, active_path, img_size)
        backend = ORTBackend(active_path, device, args.threads, data['nc'], strides)
    if args.rect and any(isinstance(dim, int) for dim in backend.input.shape[2:]):
        raise ValueError('Training rectangular validation requires dynamic H/W. Add --export-dynamic '
                         'to create a separate ONNX; the supplied model is preserved.')
    if backend.static_batch and args.infer_batch_size > backend.static_batch:
        raise ValueError(f'ONNX batch={backend.static_batch}; set --infer-batch-size accordingly')

    hyp = dict(Config.fromfile(args.config_file).data_aug) if args.config_file else {}
    data['val'], data['anno_path'] = str(source), str(annotation_path)
    dataset = TrainValDataset(
        str(source), img_size, args.batch_size, stride=max(strides), hyp=hyp,
        rect=args.rect, pad=0.5 if args.rect else 0.0,
        data_dict=data, task='val', augment=False)
    if set(map(lambda p: str(Path(p).resolve()), dataset.img_paths)) != set(image_ids):
        raise ValueError('Loader images and COCO images differ; provide an exact validation folder/JSON pair')
    # Evaler uses stems internally. Remap these to annotation IDs after its exact
    # coordinate scaling, clipping and bbox/score rounding (3/5 decimal places).
    stem_ids = {Path(path).stem: image_ids[str(Path(path).resolve())] for path in dataset.img_paths}
    if len(stem_ids) != len(dataset):
        raise ValueError('Native Evaler requires unique image stems')
    converter = Evaler(data, device=device, half=False)
    converter.is_coco = False
    predictions = {'pytorch': [], 'onnxruntime': []}
    elapsed = {name: {'forward_decode_seconds': 0.0, 'nms_restore_seconds': 0.0}
               for name in predictions}
    difference, shapes_seen, evaluated_ids = PredictionDifference(), Counter(), []
    count = min(args.limit or len(dataset), len(dataset))
    # Preserve the training loader's rect group boundaries, while loading small
    # inference batches. Reading a full 256-image group is unnecessary for a
    # static batch-1 ONNX, and also delays short --limit checks considerably.
    batches = []
    for group in range(0, count, args.batch_size):
        end = min(group + args.batch_size, count)
        for start in range(group, end, args.infer_batch_size):
            batches.append(list(range(start, min(start + args.infer_batch_size, end))))
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches,
                                        num_workers=args.workers,
                                        collate_fn=TrainValDataset.collate_fn,
                                        pin_memory=device.type == 'cuda')
    first_batch, data_seconds = True, 0.0
    progress = tqdm(total=count, desc='Shared validation')
    iterator = iter(loader)
    while len(evaluated_ids) < count:
        start = time.perf_counter()
        images, _, paths, shapes = next(iterator)
        images = images[:count - len(evaluated_ids)].to(device).float() / 255
        synchronize(device)
        data_seconds += time.perf_counter() - start
        for offset in range(0, len(images), args.infer_batch_size):
            batch = images[offset:offset + args.infer_batch_size]
            paths_batch = paths[offset:offset + len(batch)]
            shapes_batch = shapes[offset:offset + len(batch)]
            execution_batch = batch
            if backend.static_batch and len(batch) < backend.static_batch:
                execution_batch = torch.cat((batch, batch[-1:].expand(
                    backend.static_batch - len(batch), -1, -1, -1)))
            if first_batch:
                for _ in range(args.warmup):
                    model(execution_batch)
                    backend(execution_batch)
                synchronize(device)
                first_batch = False
            outputs = {}
            for name in predictions:
                start = time.perf_counter()
                prediction = (model(execution_batch)[0] if name == 'pytorch'
                              else backend(execution_batch))
                outputs[name] = prediction[:len(batch)]
                synchronize(device)
                elapsed[name]['forward_decode_seconds'] += time.perf_counter() - start
            difference.update(outputs['pytorch'], outputs['onnxruntime'])
            for name, output in outputs.items():
                start = time.perf_counter()
                detections = non_max_suppression(output, args.conf_thres, args.iou_thres,
                                                 multi_label=True, max_det=args.max_det)
                entries = converter.convert_to_coco_format(detections, batch, paths_batch,
                                                           shapes_batch, category_ids)
                for entry in entries:
                    entry['image_id'] = stem_ids[entry['image_id']]
                predictions[name].extend(entries)
                synchronize(device)
                elapsed[name]['nms_restore_seconds'] += time.perf_counter() - start
            evaluated_ids.extend(image_ids[str(Path(path).resolve())] for path in paths_batch)
            shapes_seen[f'{batch.shape[2]}x{batch.shape[3]}'] += len(batch)
            progress.update(len(batch))
    progress.close()

    results = {}
    for name, entries in predictions.items():
        (output_dir / f'{name}_predictions.json').write_text(json.dumps(entries))
        print(f'\n{name}: {len(evaluated_ids)} images, {len(entries)} detections')
        metrics = evaluate_coco(annotation_path, entries, evaluated_ids, category_ids)
        timing = {key.replace('_seconds', '_ms_per_image'): value * 1000 / count
                  for key, value in elapsed[name].items()}
        timing['forward_decode_nms_restore_ms_per_image'] = sum(timing.values())
        results[name] = {'metrics': metrics, 'detections': len(entries), 'timing': timing}
    delta = {key: results['onnxruntime']['metrics'][key] - value
             for key, value in results['pytorch']['metrics'].items()}
    report = {
        'original_onnx': file_info(original), 'evaluated_onnx': file_info(active_path),
        'weights': file_info(weights), 'data_yaml': str(Path(args.data).resolve()),
        'source': str(source), 'annotations': file_info(annotation_path),
        'settings': {'img_size': img_size, 'rect': args.rect, 'pad': 0.5 if args.rect else 0.0,
                     'validation_group_batch_size': args.batch_size,
                     'infer_batch_size': args.infer_batch_size, 'input_shape_image_counts': dict(shapes_seen),
                     'conf_thres': args.conf_thres, 'iou_thres': args.iou_thres,
                     'max_det': args.max_det, 'multi_label': True, 'dtype': 'float32',
                     'category_ids': category_ids, 'images': count, 'full_dataset': count == len(dataset),
                     'config_file': args.config_file, 'loader_hyp': hyp,
                     'warmup': args.warmup, 'threads': args.threads, 'workers': args.workers},
        'runtime': {'device': str(device), 'platform': platform.platform(),
                    'torch': torch.__version__, 'onnxruntime': ort.__version__,
                    'providers': backend.session.get_providers(), 'opencv': cv2.__version__,
                    'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
        'shared_data_wait_normalize_ms_per_image': data_seconds * 1000 / count,
        'timing_scope': 'Warmup excluded. ORT forward/decode includes CPU input/output copies and raw '
                        'head decoding; PyTorch forward includes native decoding. NMS and restoration '
                        'use the same code/device. DataLoader resize/I/O may overlap through workers. '
                        'COCO evaluation and file writing are excluded; this is a serial comparison, '
                        'not an isolated throughput benchmark.',
        'pre_nms_difference': difference.result(), 'results': results,
        'metric_delta_onnx_minus_pytorch': delta,
    }
    (output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('\nBackend          AP50        AP50:95     forward+decode   NMS+restore (ms/image)')
    for name, result in results.items():
        print(f"{name:15s}  {result['metrics']['mAP_0.50']:.8f}  "
              f"{result['metrics']['mAP_0.50_0.95']:.8f}  "
              f"{result['timing']['forward_decode_ms_per_image']:14.3f}  "
              f"{result['timing']['nms_restore_ms_per_image']:10.3f}")
    print(f'Delta (ORT - PyTorch): AP50={delta["mAP_0.50"]:+.8f}, '
          f'AP50:95={delta["mAP_0.50_0.95"]:+.8f}')
    print(f'Report: {output_dir / "comparison.json"}')


if __name__ == '__main__':
    compare(parse_args())
