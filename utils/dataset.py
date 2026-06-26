"""Dataset loading and augmentation for skin disease classification."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAIN_AUGMENTATION_PROFILES = ("strong", "noise_robust", "light")


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = x.clone()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return mixed_x, y, y[index], lam


def rand_bbox(size, lam):
    width, height = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_rat), int(height * cut_rat)

    cx, cy = np.random.randint(width), np.random.randint(height)
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(width, cx + cut_w // 2)
    bby2 = min(height, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix_data(x, y, mixup_prob=0.5, mixup_alpha=0.2, cutmix_alpha=1.0):
    if np.random.rand() < mixup_prob:
        return mixup_data(x, y, mixup_alpha)
    return cutmix_data(x, y, cutmix_alpha)


def _normalize_augmentation_profile(profile):
    return str(profile or "strong").strip().lower().replace("-", "_")


def build_train_transform(profile="strong"):
    profile = _normalize_augmentation_profile(profile)

    if profile == "strong":
        return transforms.Compose([
            transforms.Resize(320, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(
                300,
                scale=(0.75, 1.0),
                ratio=(0.8, 1.25),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandAugment(num_ops=2, magnitude=5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
        ])

    if profile == "noise_robust":
        return transforms.Compose([
            transforms.Resize(340, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(
                300,
                scale=(0.82, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.12,
                        contrast=0.12,
                        saturation=0.08,
                        hue=0.01,
                    )
                ],
                p=0.45,
            ),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.05, scale=(0.01, 0.04), ratio=(0.5, 2.0), value="random"),
        ])

    if profile == "light":
        return transforms.Compose([
            transforms.Resize(320, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(
                300,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.08,
                        contrast=0.08,
                        saturation=0.05,
                        hue=0.005,
                    )
                ],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    supported = ", ".join(TRAIN_AUGMENTATION_PROFILES)
    raise ValueError(f"Unsupported train augmentation profile: {profile}. Supported values: {supported}")


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(320, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


train_transform = build_train_transform("strong")
val_transform = build_val_transform()


def get_train_dataloader(args):
    train_aug = _normalize_augmentation_profile(getattr(args, "train_aug", "strong"))
    dataset = ImageFolder(root=args.datapath_train, transform=build_train_transform(train_aug))
    print(f"train_augmentation: {train_aug}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def get_val_dataloader(args):
    if args.val:
        dataset = ImageFolder(root=args.datapath_val, transform=val_transform)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    return None


if __name__ == "__main__":
    from utils.arguments import parse

    args = parse()
    dataloader = get_train_dataloader(args)
    print(len(dataloader))
    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)
        break
