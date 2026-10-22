# Pico small CIoU baseline with RTMDet DynamicSoftLabelAssigner.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, model, solver

amp = False

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)

solver['lr_scale'] = False
model['head']['assigner_type'] = 'dsl'
model['head']['assigner_kwargs'] = dict(
    soft_center_radius=3.0,
    topk=9,
    iou_weight=3.0,
)
model['head']['atss_warmup_epoch'] = 0  # Use DSL from the first epoch.

del deepcopy
