import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from torch import autocast

from utils.dataset import mixup_cutmix_data
from utils.lr_policy import LR


class tra_val(object):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scaler,
        args,
        train_loader,
        val_loader,
        writer,
        device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.device = device
        self.use_amp = bool(getattr(args, "amp", False) and device.type == "cuda")

        self.loss_vector = []
        self.acc_vector = []
        self.train_loss = 0
        self.total_loss = 0
        self.step = 0

        self.TrainingLoss, self.TrainingTop1, self.TrainingTop5 = ([] for _ in range(3))
        self.TestLoss, self.TestTop1, self.TestTop5 = ([] for _ in range(3))
        self.Learning_Rate = [self.args.lr]

        steps_per_epoch = len(train_loader) if train_loader else (len(val_loader) if val_loader else 100)
        self.lr_policy = LR(
            base_lr=self.args.lr,
            warmup_epoch=5,
            epochs=self.args.epochs,
            steps_per_epoch=steps_per_epoch,
        )

    def train(self, epoch):
        self.epoch = epoch
        self.losses_tr = AverageMeter()
        self.top1_tr = AverageMeter()
        self.top5_tr = AverageMeter()

        self.model.train()
        epoch_start = time.time()
        total_step = len(self.train_loader)
        log_interval = max(1, int(getattr(self.args, "log_interval", 100)))

        for i, (input, target) in enumerate(self.train_loader, start=1):
            self.lr = self.lr_policy.apply_lr(epoch, i - 1)
            self.assign_learning_rate(self.lr)

            input, target = self._move_batch(input, target)

            use_mixup = False
            lam = 1.0
            target_a = target
            target_b = target
            if np.random.rand() < 0.5:
                input, target_a, target_b, lam = mixup_cutmix_data(input, target)
                use_mixup = True

            amp_context = autocast(device_type="cuda") if self.use_amp else nullcontext()
            with amp_context:
                output = self.model(input)
                if use_mixup:
                    loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
                else:
                    loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.step += 1

            self.prec1_tr, self.prec5_tr = self.accuracy(output, target, topk=(1, 5))
            self.losses_tr.update(loss.item(), input.size(0))
            self.top1_tr.update(self.prec1_tr.item(), input.size(0))
            self.top5_tr.update(self.prec5_tr.item(), input.size(0))
            self.loss = loss
            self.write_net_values(train=True)

            if i % log_interval == 0 or i == total_step:
                print(
                    f"{datetime.now()} Epoch [{epoch + 1:03d}/{self.args.epochs:03d}], "
                    f"Step [{i:04d}/{total_step:04d}], LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                    f"Loss: {self.losses_tr.avg:.4f}, Top1: {self.top1_tr.avg:.2f}, Top5: {self.top5_tr.avg:.2f}"
                )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch + 1:03d}/{self.args.epochs:03d}] Train Summary: "
            f"Loss: {self.losses_tr.avg:.4f}, Top1: {self.top1_tr.avg:.2f}, "
            f"Top5: {self.top5_tr.avg:.2f}, Time: {elapsed:.2f}s"
        )

    def validation(self, epoch):
        self.epoch = epoch
        self.batch_time_ts = AverageMeter()
        self.losses_ts = AverageMeter()
        self.top1_ts = AverageMeter()
        self.top5_ts = AverageMeter()

        self.model.eval()
        epoch_start = time.time()

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = self._move_batch(input, target)
                output = self.model(input)
                loss = self.criterion(output, target)

                self.prec1_ts, self.prec5_ts = self.accuracy(output, target, topk=(1, 5))
                self.loss_ts = loss

                self.losses_ts.update(self.loss_ts.item(), input.size(0))
                self.top1_ts.update(self.prec1_ts.item(), input.size(0))
                self.top5_ts.update(self.prec5_ts.item(), input.size(0))
                self.write_net_values(train=False)

        elapsed = time.time() - epoch_start
        print(
            f"{datetime.now()} Epoch [{epoch + 1:03d}/{self.args.epochs:03d}] Validation: "
            f"Loss: {self.losses_ts.avg:.4f}, Top1: {self.top1_ts.avg:.2f}, "
            f"Top5: {self.top5_ts.avg:.2f}, Time: {elapsed:.2f}s"
        )
        return self.losses_ts.avg, self.top1_ts.avg, self.top5_ts.avg

    def write_net_values(self, train):
        if self.writer is None:
            return
        if train:
            self.writer.add_scalar("Loss/Training", self.loss.item(), self.step)
            self.writer.add_scalar("Top1/Training", self.prec1_tr.item(), self.step)
            self.writer.add_scalar("Optim/lr", self.lr, self.step)
            self.writer.add_scalar("Top5/Training", self.prec5_tr.data.item(), self.step)

            self.TrainingLoss.append(self.loss.item())
            self.TrainingTop1.append(self.prec1_tr.item())
            self.TrainingTop5.append(self.prec5_tr.item())
        else:
            self.writer.add_scalar("Loss/Test", self.losses_ts.avg, self.epoch + 1)
            self.writer.add_scalar("Top1/Test", self.top1_ts.avg, self.epoch + 1)
            self.writer.add_scalar("Top5/Test", self.top5_ts.avg, self.epoch + 1)

            self.TestLoss.append(self.losses_ts.avg)
            self.TestTop1.append(self.top1_ts.avg)
            self.TestTop5.append(self.top5_ts.avg)

    def assign_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
