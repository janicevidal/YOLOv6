"""COCO person baseline with normalized log-area weighting of positive losses."""

from copy import deepcopy

from configs.pico_finetune.yolov6_pico_coco_finetune_anchor_resume import (
    amp, data_aug, model, solver,
)

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)

model['head']['area_weight'] = dict(
    enabled=True,
    power=1.0,
    apply_to=('class', 'iou', 'dfl'),
)

del deepcopy
