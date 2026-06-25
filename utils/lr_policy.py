import numpy as np


class LR(object):
    def __init__(
        self,
        base_lr,
        warmup_epoch,
        epochs,
        steps_per_epoch,
        min_lr=1e-6,
        policy="cosine_lr",
        lr_steps=None,
        lr_gamma=0.1,
    ):
        self.base_lr = base_lr
        self.warmup_epoch = max(0, int(warmup_epoch))
        self.epochs = int(epochs)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.min_lr = min_lr
        self.policy = self._normalize_policy(policy)
        self.lr_steps = sorted(int(step) for step in (lr_steps or []))
        self.lr_gamma = lr_gamma
        self.total_steps = max(1, self.epochs * self.steps_per_epoch)
        self.warmup_steps = self.warmup_epoch * self.steps_per_epoch
        self.lr = base_lr
        self.current_lr = base_lr

    @staticmethod
    def _normalize_policy(policy):
        policy = str(policy or "cosine_lr").strip().lower().replace("-", "_")
        aliases = {
            "cosine": "cosine",
            "cosine_lr": "cosine",
            "cosine_annealing": "cosine",
            "cosine_annealing_lr": "cosine",
            "multistep": "multistep",
            "multi_step": "multistep",
            "multistep_lr": "multistep",
            "multi_step_lr": "multistep",
            "constant": "constant",
            "constant_lr": "constant",
            "fixed": "constant",
            "fixed_lr": "constant",
            "none": "constant",
        }
        if policy not in aliases:
            supported = ", ".join(sorted(aliases))
            raise ValueError(f"Unsupported lr_policy '{policy}'. Supported values: {supported}")
        return aliases[policy]

    def _current_step(self, epoch, step_in_epoch):
        return epoch * self.steps_per_epoch + step_in_epoch

    def warmup_lr(self, epoch, step_in_epoch):
        if self.warmup_steps <= 0:
            return self.base_lr

        current_step = self._current_step(epoch, step_in_epoch)
        return self.base_lr * current_step / self.warmup_steps

    def cosine_lr(self, epoch, step_in_epoch):
        current_step = self._current_step(epoch, step_in_epoch)
        cosine_steps = max(1, self.total_steps - self.warmup_steps - 1)
        progress = (current_step - self.warmup_steps) / cosine_steps
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    def multistep_lr(self, epoch):
        decay_count = sum(epoch >= step for step in self.lr_steps)
        return max(self.min_lr, self.base_lr * (self.lr_gamma ** decay_count))

    def apply_lr(self, epoch, step_in_epoch=0):
        if self.warmup_steps > 0 and epoch < self.warmup_epoch:
            self.current_lr = self.warmup_lr(epoch, step_in_epoch)
        elif self.policy == "cosine":
            self.current_lr = self.cosine_lr(epoch, step_in_epoch)
        elif self.policy == "multistep":
            self.current_lr = self.multistep_lr(epoch)
        else:
            self.current_lr = self.base_lr

        return self.current_lr



