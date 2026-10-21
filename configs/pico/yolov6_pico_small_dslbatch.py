# Pico small CIoU with MMYOLO DSL adapted directly to YOLOv6's TAL interface.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['head']['assigner_type'] = 'batch_dsl'
model['head']['assigner_kwargs'] = dict(
    soft_center_radius=3.0,
    topk=9,
    iou_weight=3.0,
    batch_iou=True,
)
model['head']['atss_warmup_epoch'] = 0

del deepcopy
