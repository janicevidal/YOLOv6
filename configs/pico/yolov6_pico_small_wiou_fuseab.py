# Small Pico WIoU with AF/AB joint training; inference uses only the AF head.
from copy import deepcopy

from configs.pico.yolov6_pico_small_wiou import data_aug, model, solver

amp = False

model = deepcopy(model)
solver = deepcopy(solver)
data_aug = deepcopy(data_aug)
model['head'].update(
    fuse_ab=True,
    # 416px heuristic: keep 10x13, taper scaling to 416/640 at the largest
    # anchor by log area; preserve aspect ratios before rounding. Not clustered.
    anchors_init=[[10, 13, 18, 18, 30, 21],
                  [25, 52, 48, 48, 46, 93],
                  [88, 68, 130, 130, 242, 212]],
    tal_topk_ab=26,
    ab_loss_weight=1.0,
)

del deepcopy
