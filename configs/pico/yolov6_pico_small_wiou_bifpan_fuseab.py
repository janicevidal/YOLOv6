# Lightweight PicoBiFPAN with the same AF/AB settings as small WIoU Pico.
from copy import deepcopy

from configs.pico.yolov6_pico_small_wiou_fuseab import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['backbone']['out_p2'] = True
model['neck']['type'] = 'PicoBiFPANNeck'
model['neck']['in_channels'] = [20, 40, 80, 160]
model['neck']['upsample_type'] = 'nearest'
model['neck']['activation'] = 'LeakyReLU'

del deepcopy
