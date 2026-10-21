#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import math

import torch
import torch.nn as nn

from yolov6.solver.lr_scheduler import WarmupDecayMultiCosineLR


def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)
        # Learnable scalars such as Pico's residual gates have neither name.
        # Keep them in the no-weight-decay group with normalization weights.
        for name, parameter in v.named_parameters(recurse=False):
            if name not in ('weight', 'bias') and parameter.requires_grad:
                g_bnw.append(parameter)

    optim = cfg.solver.optim
    if optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif optim in ('Adam', 'AdamW'):
        optimizer_class = torch.optim.Adam if optim == 'Adam' else torch.optim.AdamW
        default_betas = (cfg.solver.momentum, 0.999) if optim == 'Adam' else (0.9, 0.999)
        optimizer = optimizer_class(
            g_bnw,
            lr=cfg.solver.lr0,
            betas=tuple(cfg.solver.get('betas', default_betas)),
            eps=cfg.solver.get('eps', 1e-8),
            # AdamW otherwise defaults to decay on BN weights, gates and biases.
            weight_decay=0.0,
        )
    else:
        raise ValueError(f'Unknown optimizer {optim!r}; choose SGD, Adam or AdamW')

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer


def build_lr_scheduler(cfg, optimizer, epochs, steps_per_epoch=None):
    """Build learning rate scheduler from cfg file."""
    if cfg.solver.lr_scheduler == 'WarmupDecayMultiCosineLR':
        if steps_per_epoch is None or steps_per_epoch <= 0:
            raise ValueError('WarmupDecayMultiCosineLR requires positive steps_per_epoch')
        solver = cfg.solver
        warmup_iters = solver.get('warmup_iters', round(solver.get('warmup_epochs', 0) * steps_per_epoch))
        delay_iters = solver.get('delay_iters', round(solver.get('delay_epochs', warmup_iters / steps_per_epoch) * steps_per_epoch))
        decay_steps = solver.get('decay_steps')
        if decay_steps is None:
            decay_steps = [round(epoch * steps_per_epoch) for epoch in solver.get('decay_epochs', [])]
        scheduler = WarmupDecayMultiCosineLR(
            optimizer,
            max_iters=epochs * steps_per_epoch,
            warmup_iters=warmup_iters,
            warmup_factor=solver.get('warmup_factor', 0.001),
            delay_iters=delay_iters,
            decay_steps=decay_steps,
            peak_scale=solver.get('peak_scale', 0.8),
            eta_min=solver.get('eta_min', 0.05),
        )
        return scheduler, None
    elif cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        raise ValueError(f'Unknown lr scheduler {cfg.solver.lr_scheduler!r}')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf
