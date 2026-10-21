# Pico small CIoU with YOLO26 small-target-aware task-aligned assignment.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['head']['assigner_type'] = 'stal'
model['head']['assigner_kwargs'] = dict(
    topk=5,
    topk2=None,
    alpha=0.5,
    beta=6.0,
    use_ciou=True,
)
model['head']['atss_warmup_epoch'] = 0

del deepcopy
