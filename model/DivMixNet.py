"""EfficientNet-B0 + 残差注意力的 DivideMix 兼容分类模型。"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ConvBNAct(nn.Module):
    """
    Args:
        in_channels: 输入特征图通道数。
        out_channels: 输出特征图通道数。
        kernel_size: 卷积核大小。
        groups: 分组卷积组数。
        activation: 是否使用 SiLU 激活函数。
    Return:
        无直接返回值，forward 返回卷积、归一化和激活后的特征图。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        """
        Args:
            in_channels: 输入特征图通道数。
            out_channels: 输出特征图通道数。
            kernel_size: 卷积核大小。
            groups: 分组卷积组数。
            activation: 是否使用 SiLU 激活函数。
        Return:
            无直接返回值。
        """

        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            卷积、归一化和激活后的特征图。
        """

        return self.block(x)


class ResidualAttentionBlock(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        reduction: 注意力掩码分支的通道压缩比例。
        trunk_ratio: 主干分支的通道压缩比例。
    Return:
        无直接返回值，forward 返回残差注意力增强后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 16, trunk_ratio: int = 4):
        """
        Args:
            channels: 输入特征图通道数。
            reduction: 注意力掩码分支的通道压缩比例。
            trunk_ratio: 主干分支的通道压缩比例。
        Return:
            无直接返回值。
        """

        super().__init__()
        trunk_channels = max(32, channels // trunk_ratio)
        mask_channels = max(16, channels // reduction)

        self.trunk = nn.Sequential(
            ConvBNAct(channels, trunk_channels, kernel_size=1),
            ConvBNAct(trunk_channels, trunk_channels, kernel_size=3, groups=trunk_channels),
            ConvBNAct(trunk_channels, channels, kernel_size=1, activation=False),
        )
        self.mask = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mask_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mask_channels, mask_channels, kernel_size=3, padding=1, groups=mask_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mask_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            残差注意力增强后的特征图。
        """

        # 1. trunk 分支提取局部判别特征。
        trunk_feature = self.trunk(x)

        # 2. mask 分支生成通道注意力权重。
        mask_weight = self.mask(x)

        # 3. 使用残差缩放，降低新增模块对预训练特征的扰动。
        attention_feature = trunk_feature * (1.0 + mask_weight)
        return x + self.scale * attention_feature


class DivideMixHead(nn.Module):
    """
    Args:
        in_features: 输入特征维度。
        num_classes: 分类类别数量。
        dropout: Dropout 概率。
    Return:
        无直接返回值，forward 返回主分类 logits，或主辅双头 logits。
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        """
        Args:
            in_features: 输入特征维度。
            num_classes: 分类类别数量。
            dropout: Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.main_classifier = nn.Linear(in_features, num_classes)
        self.aux_classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x: 池化后的图像特征，形状为 [batch_size, in_features]。
            return_aux: 是否返回辅助分类头 logits。
        Return:
            默认返回主分类 logits；return_aux=True 时返回主辅两个 logits。
        """

        # 1. 共享特征归一化，稳定噪声标签场景下的分类头训练。
        x = self.dropout(self.norm(x))
        main_logits = self.main_classifier(x)

        # 2. 辅助头供 DivideMix 或一致性训练使用，普通训练默认不启用。
        if return_aux:
            aux_logits = self.aux_classifier(x)
            return main_logits, aux_logits
        return main_logits


class DivMixNet(nn.Module):
    """
    Args:
        num_classes: 皮肤病分类类别数量。
        pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B0 权重。
        dropout: 分类头 Dropout 概率。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(
        self,
        num_classes: int = 23,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_classes: 皮肤病分类类别数量。
            pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B0 权重。
            dropout: 分类头 Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        feature_dim = 1280

        self.features = backbone.features
        self.attention = ResidualAttentionBlock(channels=feature_dim)
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(start_dim=1)
        self.head = DivideMixHead(
            in_features=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self._init_new_layers()

    def _init_new_layers(self) -> None:
        """
        Args:
            无。
        Return:
            无直接返回值。
        """

        # 只初始化新增残差注意力和分类头，避免覆盖 EfficientNet-B0 预训练权重。
        new_layers = [self.attention, self.head]
        for module in new_layers:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x: 输入图像张量，形状为 [batch_size, 3, height, width]。
            return_aux: 是否返回辅助分类头 logits。
        Return:
            默认返回分类 logits；return_aux=True 时返回主辅两个 logits。
        """

        # 1. EfficientNet-B0 提取轻量卷积特征。
        x = self.features(x)

        # 2. 残差注意力增强关键病灶区域和通道响应。
        x = self.attention(x)

        # 3. 池化后通过 DivideMix 兼容分类头输出结果。
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.head(x, return_aux=return_aux)


if __name__ == "__main__":
    model = DivMixNet(num_classes=23, pretrained=False)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    aux_output = model(sample, return_aux=True)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"输出形状: {tuple(output.shape)}")
    print(f"双头输出形状: {tuple(aux_output[0].shape)}, {tuple(aux_output[1].shape)}")
