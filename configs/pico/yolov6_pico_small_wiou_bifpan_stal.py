# Pico LiteBlocks/PAN with lightweight three-scale BiFusion at P4 and P3.
from copy import deepcopy

from configs.pico.yolov6_pico_small_wiou import data_aug, model, solver

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['backbone']['out_p2'] = True
model['neck']['type'] = 'PicoBiFPANNeck'
model['neck']['in_channels'] = [20, 40, 80, 160]  # P2/P3/P4/P5
model['neck']['upsample_type'] = 'nearest'
model['neck']['activation'] = 'LeakyReLU'

model['head']['assigner_type'] = 'stal'
model['head']['assigner_kwargs'] = dict(
    topk=5,
    topk2=None,
    alpha=1.0,
    beta=6.0,
    use_ciou=False,
)

del deepcopy
