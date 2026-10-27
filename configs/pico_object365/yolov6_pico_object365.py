from copy import deepcopy

from configs.pico.yolov6_pico_small_wiou import data_aug, model

model = deepcopy(model)
data_aug = deepcopy(data_aug)

model['backbone']['out_p2'] = True

model['neck']['type'] = 'PicoBiFPANNeck'
model['neck']['in_channels'] = [20, 40, 80, 160]  # P2/P3/P4/P5
model['neck']['upsample_type'] = 'sni'
model['neck']['activation'] = 'LeakyReLU'
model['neck']['use_refine'] = True
model['neck']['refine_kernel_sizes'] = (5, 3)  # P3 stride=8, P4 stride=16.

model['head']['assigner_type'] = 'stal'
model['head']['assigner_kwargs'] = dict(
    topk2=None,
    alpha=1.0,
    beta=6.0,
    use_ciou=True,
)

model['head']['atss_warmup_epoch'] = 3

model['head'].update(
    fuse_ab=True,
    tal_topk=5,
    tal_topk_ab=15,
    ab_loss_weight=1.0,
    anchors_init=[
        [10, 13, 18, 18, 30, 21],       # P3 /8
        [25, 52, 48, 48, 46, 93],      # P4 /16
        [88, 68, 130, 130, 242, 212],   # P5 /32
    ],
)

solver = dict(
    optim='AdamW',
    lr0=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    warmup_bias_lr=0.0,
    lr_scheduler='WarmupDecayMultiCosineLR',
    warmup_epochs=1.0,
    warmup_factor=0.001,
    delay_epochs=75,
    decay_epochs=[150],
    peak_scale=1.0,
    eta_min=0.00001
)

del deepcopy
