#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.models.heads.effidehead_pico import Detect as PicoDetect
from yolov6.layers.common import *
from yolov6.layers.reparam import PicoBlock
from yolov6.models.pico_neck import DySample
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.general import check_version
from io import BytesIO


def prepare_model_for_export(model, raw_head=False):
    """Convert deployment blocks and select decoded or raw Pico outputs."""
    is_pico = isinstance(getattr(model, 'detect', None), PicoDetect)
    if raw_head and not is_pico:
        raise ValueError('--raw-head currently supports PicoHead only')
    model.eval()
    if is_pico:
        # Model.export controls the outer return value; Detect.export skips
        # anchor generation and LTRB decoding inside the detection head.
        model.export = raw_head
        model.detect.export = raw_head
    for layer in list(model.modules()):
        if isinstance(layer, (RepVGGBlock, PicoBlock)):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
    return model


def select_opset(model, requested=None):
    """DySample needs standard ONNX GridSample, introduced in opset 16."""
    uses_dysample = any(isinstance(layer, DySample) for layer in model.modules())
    opset = requested if requested is not None else (16 if uses_dysample else 13)
    if uses_dysample and opset < 16:
        raise ValueError('PicoNeck/DySample requires --opset 16 or newer for GridSample')
    if uses_dysample and not check_version(torch.__version__, '1.13.0'):
        raise RuntimeError(
            'Full Pico ONNX export requires PyTorch >= 1.13 and opset >= 16 '
            'for GridSample. Use a separate export environment.')
    return opset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--output', type=str, default=None, help='output ONNX path (default: beside weights)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size, the order is: height width')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--opset', type=int, default=None, help='ONNX opset (default: 16 for DySample, otherwise 13)')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--end2end', action='store_true', help='export end2end onnx')
    output_group.add_argument('--raw-head', action='store_true',
                              help='export Pico P3/P4/P5 scores and LTRB distances without anchors or decoding')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--ort', action='store_true', help='export onnx for onnxruntime')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    if args.end2end:
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
    elif args.raw_head:
        output_names = ['p3', 'p4', 'p5']
    else:
        output_names = ['outputs']
    opset = select_opset(model, args.opset)
    model = prepare_model_for_export(model, raw_head=args.raw_head)
    LOGGER.info(f'Exporting with ONNX opset {opset}')
    if args.raw_head:
        LOGGER.info('Raw Pico outputs: p3/p4/p5 [B,nc+4,H,W], channels=[class probabilities,l,t,r,b]. '
                    'Decode boxes and run NMS in the deployment postprocessor.')
    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if args.half:
        img, model = img.half(), model.half()  # to FP16
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, ConvModule):  # assign export-friendly activations
            if hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace
    dynamic_axes = None
    if args.dynamic_batch:
        args.batch_size = 'batch'
        dynamic_axes = {
            'images' :{
                0:'batch',
            },}
        output_axes = {name: {0: 'batch'} for name in output_names}
        dynamic_axes.update(output_axes)


    if args.end2end:
        from yolov6.models.end2end import End2End
        model = End2End(model, max_obj=args.topk_all, iou_thres=args.iou_thres,score_thres=args.conf_thres,
                        device=device, ort=args.ort, trt_version=args.trt_version, with_preprocess=args.with_preprocess)

    print("===================")
    print(model)
    print("===================")

    y = model(img)  # dry run

    # ONNX export
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = args.output or (os.path.splitext(args.weights)[0] + '.onnx')
        with BytesIO() as f:
            torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=['images'],
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)
            f.seek(0)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            # Fix output shape
            if args.end2end and not args.ort:
                shapes = [args.batch_size, 1, args.batch_size, args.topk_all, 4,
                          args.batch_size, args.topk_all, args.batch_size, args.topk_all]
                for i in onnx_model.graph.output:
                    for j in i.type.tensor_type.shape.dim:
                        j.dim_param = str(shapes.pop(0))
        if args.simplify:
            try:
                import onnxsim
                LOGGER.info('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                LOGGER.info(f'Simplifier failure: {e}')
        os.makedirs(os.path.dirname(os.path.abspath(export_file)), exist_ok=True)
        onnx.save(onnx_model, export_file)
        LOGGER.info(f'ONNX export success, saved as {export_file}')
    except Exception as e:
        LOGGER.error(f'ONNX export failure: {e}')
        raise

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
    if args.end2end:
        if not args.ort:
            info = f'trtexec --onnx={export_file} --saveEngine={export_file.replace(".onnx",".engine")}'
            if args.dynamic_batch:
                LOGGER.info('Dynamic batch export should define min/opt/max batchsize\n'+
                            'We set min/opt/max = 1/16/32 default!')
                wandh = 'x'.join(list(map(str,args.img_size)))
                info += (f' --minShapes=images:1x3x{wandh}'+
                f' --optShapes=images:16x3x{wandh}'+
                f' --maxShapes=images:32x3x{wandh}'+
                f' --shapes=images:16x3x{wandh}')
            LOGGER.info('\nYou can export tensorrt engine use trtexec tools.\nCommand is:')
            LOGGER.info(info)
