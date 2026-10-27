#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import shutil
import torch
import os.path as osp
from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import fuse_model


def load_state_dict(weights, model, map_location=None, exclude_keys=()):
    """Load matching checkpoint tensors, preserving exact names in exclude_keys."""
    if isinstance(exclude_keys, str):
        exclude_keys = (exclude_keys,)
    exclude_keys = set(exclude_keys or ())
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    excluded = sorted(exclude_keys.intersection(state_dict))
    if excluded:
        LOGGER.info(f'Excluded pretrained keys (keeping model initialization): {excluded}')
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in exclude_keys and k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(filename, best_filename)


def strip_optimizer(ckpt_dir, epoch, save_fp32=False):
    """Delete optimizer state and save model weights in the requested precision."""
    for s in ['best', 'last']:
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pt'.format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if ckpt.get('ema'):
            ckpt['model'] = ckpt['ema']  # replace model with ema
        for k in ['optimizer', 'ema', 'updates']:  # keys
            ckpt[k] = None
        ckpt['epoch'] = epoch
        dtype = torch.float32 if save_fp32 else torch.float16
        ckpt['model'].to(dtype=dtype)
        for p in ckpt['model'].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)
