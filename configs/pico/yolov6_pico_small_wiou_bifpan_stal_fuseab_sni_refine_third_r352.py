model = dict(
    type='YOLOv6pico',
    pretrained='/home/zhangxiaoshuai/Checkpoint/HumanDetetct/checkpoints/yolov6_pico_small_wiou_bifpan_stal_fuseab_sni_refine_resume_r352/weights/last_ckpt.pt',
    backbone=dict(
        type='PicoBackbone',
        stem_channels=20,
        stage_channels=(40, 80, 160),
        stage_depths=(2, 4, 2),
        out_stages=(0, 1, 2),
        out_p2=True,
    ),
    neck=dict(
        type='PicoBiFPANNeck',
        in_channels=[20, 40, 80, 160],
        out_channels=48,
        upsample_type='sni',  # 'nearest', 'sni' or 'dysample'
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        use_depthwise=True,
        activation='LeakyReLU',
        use_refine=True,
        refine_kernel_sizes=(5, 3)
    ),
    head=dict(
        type='PicoHead',
        in_channels=[48, 48, 48],
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='wiou',
        iou_kwargs= dict(
            wiou_monotonous=False,
            wiou_momentum=1 - 0.5 ** (1 / 7000),
            wiou_gamma=1.9,
            wiou_delta=3.0,
        ),
        use_dfl=False,
        reg_max=0,
        assigner_type='stal',
        assigner_kwargs=dict(
            topk2=None,
            alpha=1.0,
            beta=6.0,
            use_ciou=True
        ),
        fuse_ab=True,
        tal_topk=5,
        tal_topk_ab=15,
        ab_loss_weight=1.0,
        anchors_init=[
            [10, 13, 18, 18, 30, 21],       # P3 /8
            [25, 52, 48, 48, 46, 93],      # P4 /16
            [88, 68, 130, 130, 242, 212],   # P5 /32
        ],
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