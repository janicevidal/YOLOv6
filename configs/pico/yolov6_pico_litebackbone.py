# Backbone comparison: FlashDet LiteBackbone-0.5x + unchanged PicoNeck/PicoHead.
from copy import deepcopy

from configs.pico.yolov6_pico import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['backbone'] = dict(
    type='LiteBackbone',
    model_size='0.5x',
    out_stages=(2, 3, 4),
    pretrained=True,
    activation='LeakyReLU',
)

del deepcopy
