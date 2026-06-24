import os

import torch


class OutputSave(object):
    def __init__(self, model, args, optimizer):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.best_top1 = 0.0
        self.best_top5 = 0.0
        self.no_improve_epochs = 0

    def save_checkpoint(self, epoch):
        root = self.args.save_path
        os.makedirs(root, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_top1": self.best_top1,
            "best_top5": self.best_top5,
            "no_improve_epochs": self.no_improve_epochs,
        }
        torch.save(checkpoint, os.path.join(root, "checkpoint.pth.tar"))

    def update_best(self, top1, top5, epoch):
        improved = top1 > self.best_top1
        previous_top1 = self.best_top1
        self.best_top1 = max(self.best_top1, top1)
        self.best_top5 = max(self.best_top5, top5)

        if not improved:
            return False

        root = self.args.save_path
        os.makedirs(root, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_top1": self.best_top1,
            "best_top5": self.best_top5,
            "no_improve_epochs": self.no_improve_epochs,
        }
        torch.save(checkpoint, os.path.join(root, "best_model.pth.tar"))
        print(
            f"### Best Model Saved (Top1 improved from {previous_top1:.4f} "
            f"to {self.best_top1:.4f}, Top5: {self.best_top5:.4f}) ###"
        )
        return True
