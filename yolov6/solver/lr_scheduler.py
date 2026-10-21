"""Iteration-based learning-rate schedules."""

import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupDecayMultiCosineLR(_LRScheduler):
    """Linear warmup, a hold, then cosine stages with scaled peak restarts.

    All boundaries are absolute training iteration indices. ``eta_min`` is a
    ratio of the current stage's peak, not an absolute learning rate. Each
    milestone belongs to the stage ending there; the following iteration starts
    the next stage. After the final milestone, hold that stage's minimum.
    """

    by_iteration = True

    def __init__(self, optimizer, max_iters, warmup_iters=0, warmup_factor=0.001,
                 delay_iters=0, decay_steps=None, peak_scale=0.8, eta_min=0.05,
                 last_epoch=-1):
        for name, value in (('max_iters', max_iters), ('warmup_iters', warmup_iters),
                            ('delay_iters', delay_iters)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f'{name} must be an integer')
        if max_iters <= 0 or not 0 <= warmup_iters <= delay_iters <= max_iters:
            raise ValueError('Require max_iters > 0 and 0 <= warmup_iters <= delay_iters <= max_iters')
        if not 0 <= warmup_factor <= 1 or not 0 <= eta_min <= 1:
            raise ValueError('warmup_factor and eta_min must be ratios in [0, 1]')
        if not math.isfinite(peak_scale) or peak_scale <= 0:
            raise ValueError('peak_scale must be finite and positive')

        milestones = []
        for step in (() if decay_steps is None else decay_steps):
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError('decay_steps must contain integer iteration indices')
            if step > delay_iters:
                milestones.append(min(step, max_iters))
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.delay_iters = delay_iters
        self.decay_steps = sorted(set(milestones)) or [max_iters]
        self.peak_scale = peak_scale
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.get_lr_at(self.last_epoch)

    def get_lr_at(self, iteration):
        """Read a planned iteration's LRs without advancing scheduler state."""
        factor = self._lr_factor(iteration)
        return [base_lr * factor for base_lr in self.base_lrs]

    def _lr_factor(self, iteration):
        iteration = min(max(iteration, 0), self.max_iters)
        if iteration < self.warmup_iters:
            alpha = iteration / self.warmup_iters
            return self.warmup_factor * (1 - alpha) + alpha
        if iteration <= self.delay_iters:
            return 1.0

        previous = self.delay_iters
        peak = 1.0
        for milestone in self.decay_steps:
            if iteration <= milestone:
                position = (iteration - previous) / (milestone - previous)
                cosine = (1 + math.cos(math.pi * position)) / 2
                return peak * (self.eta_min + (1 - self.eta_min) * cosine)
            previous = milestone
            peak *= self.peak_scale
        return (peak / self.peak_scale) * self.eta_min
