""""
分类模块的数据增强
"""

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.arguments import parse


# MixUp数据增强
def mixup_data(x, y, alpha=0.2):
    """
    x: 输入图像    shape ：[batch_size]
    y: 标签
    alpha: Beta分布参数
    """
    #1.从Beta分布中随机生成0-1之间的混合比例
    lam = np.random.beta(alpha, alpha)
    #2.获取batch大小   4
    batch_size = x.size(0)
    #3.生成打乱中的索引（用于取另一张图）  index[2,0,3,1]
    index = torch.randperm(batch_size)
    #4.混合 = 原图 * 混合比例 + 另一张图 * (1 - 混合比例)
    mixed_x = lam * x + (1 - lam) * x[index]
    #5.返回混合图、原标签、另一张图的标签、混合比例
    return mixed_x, y, y[index], lam


# CutMix数据增强
def cutmix_data(x, y, alpha=1.0):
    """
    x: 输入图像 batch
    y: 标签
    alpha: Beta分布参数
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    #计算裁剪区域的坐标，左上  右下
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    #将另一张图对应区域贴过来
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    #计算实际裁剪面积比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    #返回混合图、原标签、另一张图的标签、混合比例
    return x, y, y[index], lam


# 计算CutMix的裁剪区域
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    #随机选择中心坐标
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2


# 混合增强选择器
def mixup_cutmix_data(x, y, prob=0.5, alpha=0.2):
    """
    随机选择MixUp或CutMix
    prob: 选择MixUp的概率（0.5表示一半一半）
    """
    if np.random.rand() < prob:
        return mixup_data(x, y, alpha)
    else:
        return cutmix_data(x, y, alpha)



train_transform = transforms.Compose([
    transforms.Resize((256, 256)),          # 稍微大一点，留给裁剪空间
    transforms.RandomHorizontalFlip(),     # 水平翻转（皮肤病常对称）
    transforms.RandomVerticalFlip(p=0.3),   # 垂直翻转
    transforms.RandomRotation(45),         # 随机旋转 ±45°
    transforms.ColorJitter(
        brightness=0.3,                    # 亮度调整
        contrast=0.3,                      # 对比度调整
        saturation=0.3,                     #饱和度
        hue=0.1                             #色调，颜色的种类
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),       #平移
    transforms.RandomCrop((224, 224)),    # 随机裁剪到目标尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((256, 256)),        # 统一尺寸
    transforms.CenterCrop((224, 224)),    # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Use a project-relative Linux path.
# train_data = ImageFolder(root='./skin diseases/train-new', transform=transform)
# print(len(train_data.classes))
# print(train_data.classes)


# 获取加载器
def get_train_dataloader(args):
    datasets = ImageFolder(root=args.datapath_train, transform=train_transform)
    return DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def get_val_dataloader(args):
    if args.val:
        datasets = ImageFolder(root=args.datapath_val, transform=val_transform)
        return DataLoader(
            datasets,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    return None





if __name__ == '__main__':
    args = parse()
    dataloader = get_train_dataloader(args)
    print(len(dataloader))
    for images , labels in dataloader:
        print(images.shape)
        print(labels.shape)
        break
