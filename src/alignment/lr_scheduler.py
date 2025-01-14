import warnings

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCooldownScheduler(_LRScheduler):
    """Custom learning rate scheduler with warmup and cooldown phases.

    The scheduler has three phases:
    1. Warmup: LR increases linearly from 0 to target_lr in first 10% steps
    2. Constant: Maintains target_lr from 10% to 90% of steps
    3. Cooldown: Decreases linearly from target_lr to 0 in final 10% steps
    """

    def __init__(self, optimizer, total_steps, target_lr, last_epoch=-1):
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.warmup_steps = int(0.1 * total_steps)  # 10% of total steps
        self.cooldown_start = int(0.9 * total_steps)  # 90% of total steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        step = self.last_epoch

        # Warmup phase
        if step < self.warmup_steps:
            lr_scale = step / max(1, self.warmup_steps)
            return [self.target_lr * lr_scale for _ in self.base_lrs]

        # Constant phase
        elif step < self.cooldown_start:
            return [self.target_lr for _ in self.base_lrs]

        # Cooldown phase
        else:
            remaining_steps = self.total_steps - step
            total_cooldown_steps = self.total_steps - self.cooldown_start
            lr_scale = remaining_steps / max(1, total_cooldown_steps)
            return [self.target_lr * lr_scale for _ in self.base_lrs]
