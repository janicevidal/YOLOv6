# Pico small CIoU with soft nearest-neighbor interpolation in PicoNeck.
from copy import deepcopy

from configs.pico.yolov6_pico_small_wiou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['neck']['upsample_type'] = 'sni'

del deepcopy
