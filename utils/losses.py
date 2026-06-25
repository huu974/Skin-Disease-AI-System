"""Loss builders for skin disease classification."""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_class_counts(dataset, num_classes: int) -> list[int]:
    """Return sample counts in dataset class-index order."""
    if hasattr(dataset, "targets"):
        targets = list(dataset.targets)
    elif hasattr(dataset, "samples"):
        targets = [label for _, label in dataset.samples]
    else:
        targets = []
        for _, label in dataset:
            targets.append(int(label))

    counts = Counter(int(label) for label in targets)
    return [counts.get(class_index, 0) for class_index in range(num_classes)]


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on effective number of samples.

    Reference:
    "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
    """

    def __init__(
        self,
        samples_per_cls: list[int],
        num_classes: int,
        loss_type: str = "focal",
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smooth: float = 0.0,
    ):
        super().__init__()
        if loss_type not in {"focal", "sigmoid", "softmax", "cross_entropy"}:
            raise ValueError(f"Unsupported class-balanced loss type: {loss_type}")
        if len(samples_per_cls) != num_classes:
            raise ValueError("samples_per_cls length must match num_classes")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if not 0.0 <= label_smooth < 1.0:
            raise ValueError("label_smooth must be in [0, 1)")

        self.num_classes = num_classes
        self.loss_type = loss_type
        self.gamma = gamma
        self.label_smooth = label_smooth

        samples = torch.tensor(samples_per_cls, dtype=torch.float32)
        samples = samples.clamp_min(1.0)
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), samples)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        self.register_buffer("class_weights", weights.float())

    def forward(self, logits, labels):
        labels = labels.long()

        if self.loss_type == "cross_entropy":
            return F.cross_entropy(logits, labels, weight=self.class_weights, label_smoothing=self.label_smooth)

        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        if self.label_smooth > 0.0:
            labels_one_hot = labels_one_hot * (1.0 - self.label_smooth) + self.label_smooth / self.num_classes
        sample_weights = self.class_weights[labels].unsqueeze(1).repeat(1, self.num_classes)

        if self.loss_type == "focal":
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels_one_hot, reduction="none")
            probs = torch.sigmoid(logits)
            probs_true = labels_one_hot * probs + (1.0 - labels_one_hot) * (1.0 - probs)
            modulator = torch.pow(1.0 - probs_true, self.gamma)
            loss = sample_weights * modulator * bce_loss
            return loss.sum() / labels_one_hot.sum().clamp_min(1.0)

        if self.loss_type == "sigmoid":
            return F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=sample_weights)

        pred = torch.softmax(logits, dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(pred, labels_one_hot, weight=sample_weights)


def build_loss(args, train_dataset, num_classes: int) -> nn.Module:
    """Build the configured training loss."""
    loss_name = getattr(args, "loss", "cross_entropy")
    label_smooth = float(getattr(args, "label_smooth", 0.0))
    if not 0.0 <= label_smooth < 1.0:
        raise ValueError("--label-smooth must be in [0, 1)")

    if loss_name == "cross_entropy":
        print(f"Using loss: cross_entropy (label_smooth={label_smooth})")
        return nn.CrossEntropyLoss(label_smoothing=label_smooth)

    if loss_name == "class_balanced":
        samples_per_cls = get_class_counts(train_dataset, num_classes)
        loss_type = getattr(args, "cb_loss_type", "focal")
        beta = float(getattr(args, "cb_beta", 0.9999))
        gamma = float(getattr(args, "cb_gamma", 2.0))
        print(
            "Using loss: class_balanced "
            f"(type={loss_type}, beta={beta}, gamma={gamma}, label_smooth={label_smooth}, "
            f"samples_per_cls={samples_per_cls})"
        )
        return ClassBalancedLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            loss_type=loss_type,
            beta=beta,
            gamma=gamma,
            label_smooth=label_smooth,
        )

    raise ValueError(f"Unsupported loss: {loss_name}")
