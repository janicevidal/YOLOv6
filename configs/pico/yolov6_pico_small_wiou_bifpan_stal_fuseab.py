# PicoBiFPAN + WIoU + STAL + AF/AB joint training.
from copy import deepcopy

from configs.pico.yolov6_pico_small_wiou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)

model['backbone']['out_p2'] = True

model['neck']['type'] = 'PicoBiFPANNeck'
model['neck']['in_channels'] = [20, 40, 80, 160]  # P2/P3/P4/P5
model['neck']['upsample_type'] = 'nearest'
model['neck']['activation'] = 'LeakyReLU'

model['head']['assigner_type'] = 'stal'
model['head']['assigner_kwargs'] = dict(
    topk2=None,
    alpha=1.0,
    beta=6.0,
    use_ciou=False,
)

model['head'].update(
    fuse_ab=True,
    tal_topk=5,
    tal_topk_ab=10,
    ab_loss_weight=1.0,
    anchors_init=[
        [10, 13, 18, 18, 30, 21],       # P3 /8
        [25, 52, 48, 48, 46, 93],      # P4 /16
        [88, 68, 130, 130, 242, 212],   # P5 /32
    ],
)

del deepcopy