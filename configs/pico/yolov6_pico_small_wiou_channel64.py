# Same small Pico experiment as the CIoU configuration, using Wise-IoU v3.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, solver

model = dict(
    type='YOLOv6pico',
    pretrained=None,
    backbone=dict(
        type='PicoBackbone',
        stem_channels=20,
        stage_channels=(40, 80, 160),
        stage_depths=(2, 4, 1),
        out_stages=(0, 1, 2),
    ),
    neck=dict(
        type='PicoNeck',
        in_channels=[40, 80, 160],
        out_channels=64,
        upsample_type='nearest',  # 'nearest' or 'dysample'
        kernel_size=3,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        use_depthwise=True,
        activation='LeakyReLU',
    ),
    head=dict(
        type='PicoHead',
        in_channels=[64, 64, 64],
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        atss_warmup_epoch=10,
        tal_topk=5,
        iou_type='wiou',
        use_dfl=False,
        reg_max=0,
        iou_kwargs= dict(
            wiou_monotonous=False,  # None: v1; True: monotonic v2; False: non-monotonic v3.
            wiou_momentum=1 - 0.5 ** (1 / 7000),
            wiou_gamma=1.9,
            wiou_delta=3.0,
        )
    ),
)

solver = deepcopy(solver)
data_aug = deepcopy(data_aug)

del deepcopy
