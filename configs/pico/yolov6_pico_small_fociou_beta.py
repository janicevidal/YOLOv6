# Same small Pico experiment as the CIoU configuration, using YOLO-ULM FoCIoU.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['head']['iou_type'] = 'fociou'
model['head']['iou_kwargs'] = dict(
    fociou_m=0.0,
    fociou_n=0.95,
    fociou_beta=1.5,
)

del deepcopy
