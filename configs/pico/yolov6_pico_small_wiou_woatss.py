# Same small Pico experiment as the CIoU configuration, using Wise-IoU v3.
from copy import deepcopy

from configs.pico.yolov6_pico_small_ciou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['head']['iou_type'] = 'wiou'
model['head']['iou_kwargs'] = dict(
    wiou_monotonous=False,  # None: v1; True: monotonic v2; False: non-monotonic v3.
    wiou_momentum=1 - 0.5 ** (1 / 7000),
    wiou_gamma=1.9,
    wiou_delta=3.0,
)

model['head']['atss_warmup_epoch'] = 0

del deepcopy
