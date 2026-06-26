import torch.nn as nn


"""Loss builders for skin disease classification."""


def build_loss(args, train_dataset, num_classes: int) -> nn.Module:
    """
    Args:
        args: 训练命令行参数。
        train_dataset: 训练数据集对象，此处保留参数用于兼容调用接口。
        num_classes: 多分类类别数量，此处保留参数用于兼容调用接口。
    Return:
        多分类交叉熵损失函数。
    """

    loss_name = getattr(args, "loss", "cross_entropy")
    label_smooth = float(getattr(args, "label_smooth", 0.0))
    _ = train_dataset, num_classes

    if loss_name != "cross_entropy":
        raise ValueError("当前项目固定使用多分类交叉熵损失：cross_entropy")
    if not 0.0 <= label_smooth < 1.0:
        raise ValueError("--label-smooth must be in [0, 1)")

    print(f"Using loss: cross_entropy (label_smooth={label_smooth})")
    return nn.CrossEntropyLoss(label_smoothing=label_smooth)
