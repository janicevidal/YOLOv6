#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.models.pico_backbone import PicoBackbone
from yolov6.models.lite_backbone import LiteBackbone
from yolov6.models.pico_neck import PicoNeck
from yolov6.models.pico_bifpan import PicoBiFPANNeck
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=None, distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        x = self.backbone(x)
        x = self.neck(x)
        if not export_mode:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def _build_rep_backbone_neck(config, channels):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:

        if "stage_block_type" in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = "BepC3"  #default

        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf,
            stage_block_type=stage_block_type
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e,
            stage_block_type=stage_block_type
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    return backbone, neck, channels_list


def _build_pico_backbone_neck(config, channels, num_layers):
    backbones = {'PicoBackbone': PicoBackbone, 'LiteBackbone': LiteBackbone}
    necks = {'PicoNeck': PicoNeck, 'PicoBiFPANNeck': PicoBiFPANNeck}
    backbone_type = config.model.backbone.type
    neck_type = config.model.neck.type
    if backbone_type not in backbones or neck_type not in necks:
        raise ValueError('Configure a Pico neck with PicoBackbone or LiteBackbone')
    if num_layers != 3 or list(config.model.head.strides) != [8, 16, 32]:
        raise ValueError('Pico requires three detection levels with strides [8, 16, 32]')

    backbone_cfg = dict(config.model.backbone)
    backbone_cfg.pop('type')
    backbone = backbones[backbone_type](in_channels=channels, **backbone_cfg)
    required_stages = (0, 1, 2) if backbone_type == 'PicoBackbone' else (2, 3, 4)
    if tuple(backbone.out_stages) != required_stages:
        raise ValueError(f'{backbone_type} must output backbone stages {required_stages} for YOLOv6')
    requires_p2 = neck_type == 'PicoBiFPANNeck'
    if bool(backbone.out_p2) != requires_p2:
        raise ValueError(f'{neck_type} requires backbone.out_p2={requires_p2}')

    neck_cfg = dict(config.model.neck)
    neck_cfg.pop('type')
    neck_channels = neck_cfg.pop('in_channels', backbone.out_channels)
    if list(neck_channels) != backbone.out_channels:
        raise ValueError(f'{neck_type} input channels must match backbone outputs')
    if neck_cfg.get('num_extra_level', 0) != 0:
        raise ValueError('The three-level Pico detection head requires num_extra_level=0')
    neck = necks[neck_type](in_channels=backbone.out_channels, **neck_cfg)
    head_channels = [neck.out_channels] * num_layers
    if list(config.model.head.get('in_channels', head_channels)) != head_channels:
        raise ValueError('Detection head input channels must match PicoNeck outputs')
    return backbone, neck, head_channels


def build_network(config, channels, num_classes, num_layers, fuse_ab=None, distill_ns=False):
    if fuse_ab is None:
        fuse_ab = config.model.head.get('fuse_ab', False)
    if config.model.backbone.type in ('PicoBackbone', 'LiteBackbone') or config.model.neck.type in ('PicoNeck', 'PicoBiFPANNeck'):
        backbone, neck, channels_list = _build_pico_backbone_neck(config, channels, num_layers)
    else:
        backbone, neck, channels_list = _build_rep_backbone_neck(config, channels)
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max

    if config.model.head.type == 'PicoHead':
        if config.model.backbone.type not in ('PicoBackbone', 'LiteBackbone') or config.model.neck.type not in ('PicoNeck', 'PicoBiFPANNeck'):
            raise ValueError('PicoHead requires a Pico neck with PicoBackbone or LiteBackbone')
        if distill_ns:
            raise ValueError('PicoHead does not support distillation')
        if use_dfl or reg_max != 0:
            raise ValueError('PicoHead requires use_dfl=False and reg_max=0 for direct LTRB regression')
        from yolov6.models.heads.effidehead_pico import Detect
        head = Detect(num_classes, channels_list, config.model.head.strides,
                      fuse_ab=fuse_ab, anchors_init=config.model.head.get('anchors_init'))

    elif distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(cfg, num_classes, device, fuse_ab=None, distill_ns=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model
