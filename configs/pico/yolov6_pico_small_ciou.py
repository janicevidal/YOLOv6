# PicoBackbone (20/40/80/160, depths 2/4/2) + 48-channel PicoNeck/PicoHead.
# Single class: 218,313 parameters before deployment conversion; 222,093 after
# structural reparameterization, before further ONNX Conv/BN folding.
# Pico channels/depths are specified directly, without Rep network scaling.

model = dict(
    type='YOLOv6pico',
    pretrained=None,
    backbone=dict(
        type='PicoBackbone',
        stem_channels=20,
        stage_channels=(40, 80, 160),
        stage_depths=(2, 4, 2),
        out_stages=(0, 1, 2),
    ),
    neck=dict(
        type='PicoNeck',
        in_channels=[40, 80, 160],
        out_channels=48,
        upsample_type='nearest',  # 'nearest', 'sni' or 'dysample'
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        use_depthwise=True,
        activation='LeakyReLU',
    ),
    head=dict(
        type='PicoHead',
        in_channels=[48, 48, 48],
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        atss_warmup_epoch=10,
        tal_topk=5,
        iou_type='ciou',
        use_dfl=False,
        reg_max=0,
    ),
)

solver = dict(
    optim='AdamW',
    lr0=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    warmup_bias_lr=0.0,
    lr_scheduler='WarmupDecayMultiCosineLR',
    warmup_epochs=3.0,
    warmup_factor=0.001,
    delay_epochs=150,
    decay_epochs=[300, 400],
    peak_scale=0.5,
    eta_min=0.00001
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
)
