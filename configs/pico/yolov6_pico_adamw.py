# Pico architecture/augmentation with a configurable AdamW solver.
from copy import deepcopy

from configs.pico.yolov6_pico import data_aug, model, solver

# Isolate mutable dictionaries from the imported SGD configuration.
model = deepcopy(model)
data_aug = deepcopy(data_aug)
solver = dict(
    solver,
    optim='AdamW',
    lr0=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    warmup_bias_lr=0.0,
    lr_scheduler='WarmupDecayMultiCosineLR',
    warmup_epochs=2.0,
    warmup_factor=0.001,
    delay_epochs=200.0,
    decay_epochs=[300, 400],
    peak_scale=0.5,
    eta_min=0.00001
)

del deepcopy
