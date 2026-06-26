"""Dataset loading for skin disease classification."""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform():
    """
    Args:

    Return:
        不含随机数据增强的训练预处理。
    """
    return transforms.Compose([
        transforms.Resize(320, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(320, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


train_transform = build_train_transform()
val_transform = build_val_transform()


def get_train_dataloader(args):
    dataset = ImageFolder(root=args.datapath_train, transform=build_train_transform())
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
