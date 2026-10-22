# Pico AF head with a training-only, three-anchor AB auxiliary branch.
from copy import deepcopy

from configs.pico.yolov6_pico import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['head'].update(
    fuse_ab=True,
    anchors_init=[[10, 13, 19, 19, 33, 23],
                  [30, 61, 59, 59, 59, 119],
                  [116, 90, 185, 185, 373, 326]],
    tal_topk_ab=10,
    ab_loss_weight=1.0,
)

del deepcopy
