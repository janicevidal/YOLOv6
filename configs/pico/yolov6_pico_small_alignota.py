# Pico small CIoU baseline with DAMO-YOLO AlignOTA assignment.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, model, solver

amp = False

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)

solver['lr_scale'] = False
model['head']['assigner_type'] = 'alignota'
model['head']['assigner_kwargs'] = dict(
    center_radius=2.5,
    candidate_topk=9,
    iou_weight=3.0,
    cls_weight=1.0,
)
model['head']['atss_warmup_epoch'] = 0  # Use AlignOTA from the first epoch.

del deepcopy
