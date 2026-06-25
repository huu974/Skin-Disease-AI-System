import os
from collections import OrderedDict

import torch


class OutputSave(object):
    def __init__(self, model, args, optimizer):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.best_top1 = 0.0
        self.best_top5 = 0.0
        self.best_epoch = None
        self.no_improve_epochs = 0

    def _to_cpu(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, OrderedDict):
            return OrderedDict((key, self._to_cpu(item)) for key, item in value.items())
        if isinstance(value, dict):
            return {key: self._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu(item) for item in value)
        return value

    def _build_checkpoint(self, epoch):
        return {
            "epoch": epoch,
            "model_state_dict": self._to_cpu(self.model.state_dict()),
            "optimizer_state_dict": self._to_cpu(self.optimizer.state_dict()),
            "best_top1": self.best_top1,
            "best_top5": self.best_top5,
            "best_epoch": self.best_epoch,
            "no_improve_epochs": self.no_improve_epochs,
        }

    def _atomic_save(self, checkpoint, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def save_checkpoint(self, epoch):
        root = self.args.save_path
        os.makedirs(root, exist_ok=True)

        path = os.path.join(root, "checkpoint.pth.tar")
        self._atomic_save(self._build_checkpoint(epoch), path)
        print(f"### Checkpoint Saved: {path} (epoch {epoch + 1}) ###")

    def update_best(self, top1, top5, epoch):
        improved = top1 > self.best_top1
        previous_top1 = self.best_top1
        self.best_top1 = max(self.best_top1, top1)
        self.best_top5 = max(self.best_top5, top5)
        if improved:
            self.best_epoch = epoch + 1

        if not improved:
            return False

        root = self.args.save_path
        os.makedirs(root, exist_ok=True)
        self._atomic_save(self._build_checkpoint(epoch), os.path.join(root, "best_model.pth.tar"))
        print(
            f"### Best Model Saved (Top1 improved from {previous_top1:.4f} "
            f"to {self.best_top1:.4f}, Top5: {self.best_top5:.4f}) ###"
        )
        return True
