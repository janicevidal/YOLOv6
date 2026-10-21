"""Training diagnostic plots."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.optim.lr_scheduler import LambdaLR

from yolov6.solver.lr_scheduler import WarmupDecayMultiCosineLR
from yolov6.utils.events import LOGGER


def get_lr_schedule(optimizer, scheduler, total_iters, steps_per_epoch=1,
                    warmup_iters=0, warmup_bias_lr=0.0):
    """Read the full planned LR sequence without modifying training objects.

    Supports the repository's iteration scheduler and epoch-based LambdaLR,
    including the trainer's separate LR warmup. No model/optimizer tensors are
    copied and no optimizer or scheduler steps are executed.
    """
    if total_iters <= 0 or steps_per_epoch <= 0 or warmup_iters < 0:
        raise ValueError('Require positive total_iters/steps_per_epoch and nonnegative warmup_iters')
    if scheduler.optimizer is not optimizer:
        raise ValueError('scheduler must belong to the supplied optimizer')
    base_lrs = scheduler.base_lrs
    lrs = np.empty((total_iters, len(base_lrs)), dtype=np.float64)

    if isinstance(scheduler, WarmupDecayMultiCosineLR):
        for iteration in range(total_iters):
            lrs[iteration] = scheduler.get_lr_at(iteration)
    elif isinstance(scheduler, LambdaLR):
        current_lrs = [base * factor(0) for base, factor in zip(base_lrs, scheduler.lr_lambdas)]
        for iteration in range(total_iters):
            epoch = iteration // steps_per_epoch
            if iteration <= warmup_iters:
                current_lrs = [
                    np.interp(iteration, [0, warmup_iters],
                              [warmup_bias_lr if group == 2 else 0.0, base * factor(epoch)])
                    for group, (base, factor) in enumerate(zip(base_lrs, scheduler.lr_lambdas))
                ]
            lrs[iteration] = current_lrs
            # The trainer resets last_epoch=-1, then steps after each epoch.
            if (iteration + 1) % steps_per_epoch == 0:
                current_lrs = [base * factor(epoch) for base, factor in zip(base_lrs, scheduler.lr_lambdas)]
    else:
        raise TypeError(f'Unsupported LR scheduler for plotting: {type(scheduler).__name__}')
    return lrs


def plot_lr_scheduler(optimizer, scheduler, total_iters=100000, save_dir='.',
                      steps_per_epoch=1, warmup_iters=0, warmup_bias_lr=0.0,
                      start_iter=0):
    """Save all groups' planned LRs to lr_schedule.png, marking resume position."""
    lrs = get_lr_schedule(optimizer, scheduler, total_iters, steps_per_epoch,
                          warmup_iters, warmup_bias_lr)
    figure = Figure(figsize=(12, 5))
    FigureCanvasAgg(figure)  # Save on headless training hosts without pyplot state.
    axes = figure.subplots()
    labels = ('BN / gates', 'Weights', 'Biases') if lrs.shape[1] == 3 else (
        tuple(f'Group {group}' for group in range(lrs.shape[1])))
    styles = ('-', '--', ':')
    for group, label in enumerate(labels):
        axes.plot(np.arange(total_iters), lrs[:, group], label=label,
                  linewidth=2, linestyle=styles[group % len(styles)])
    if start_iter > 0:
        axes.axvline(start_iter, color='gray', linestyle='--', label=f'Resume at {start_iter}')
    axes.set(xlabel='Iteration', ylabel='Learning Rate', title='Learning Rate Schedule',
             xlim=(0, total_iters))
    axes.grid(True, linestyle='--', alpha=0.7)
    axes.legend()
    path = Path(save_dir) / 'lr_schedule.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=200, bbox_inches='tight')
    finally:
        figure.clear()
    LOGGER.info(f'LR curve saved to {path}')
    return path
