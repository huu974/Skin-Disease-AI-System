import numpy as np


class LR(object):
    def __init__(self,
                 base_lr,
                 warmup_epoch,
                 epochs,
                 steps_per_epoch,
                 min_lr=1e-6,
                 ):
        self.base_lr = base_lr          # 基础学习率
        self.warmup_epoch = warmup_epoch  # 预热轮数
        self.epochs = epochs            # 总训练轮数
        self.steps_per_epoch = steps_per_epoch  # 每个epoch的步数
        self.min_lr = min_lr            # 最小学习率
        self.total_steps = epochs * steps_per_epoch  # 总步数
        self.lr = base_lr
        self.current_lr = base_lr


    def warmup_lr(self, epoch, step_in_epoch):
        current_step = epoch * self.steps_per_epoch + step_in_epoch
        warmup_steps = self.warmup_epoch * self.steps_per_epoch
        return self.base_lr * current_step / warmup_steps


    def cosine_lr(self, epoch, step_in_epoch):
        current_step = epoch * self.steps_per_epoch + step_in_epoch
        progress = current_step / self.total_steps
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


    def apply_lr(self, epoch, step_in_epoch=0):
        # 预热阶段：学习率线性增长
        if epoch < self.warmup_epoch:
            self.current_lr = self.warmup_lr(epoch, step_in_epoch)
        # 余弦退火阶段：学习率平滑下降
        else:
            self.current_lr = self.cosine_lr(epoch, step_in_epoch)

        return self.current_lr




