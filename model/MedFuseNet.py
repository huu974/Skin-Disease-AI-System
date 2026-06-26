"""EfficientNet-B3 多尺度注意力融合分类模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class ConvBNAct(nn.Module):
    """
    Args:
        in_channels: 输入特征图通道数。
        out_channels: 输出特征图通道数。
        kernel_size: 卷积核大小。
        stride: 卷积步长。
        padding: 卷积填充大小。
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
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: bool = True,
    ):
        """
        Args:
            in_channels: 输入特征图通道数。
            out_channels: 输出特征图通道数。
            kernel_size: 卷积核大小。
            stride: 卷积步长。
            padding: 卷积填充大小。
            groups: 分组卷积组数。
            activation: 是否使用 SiLU 激活函数。
        Return:
            无直接返回值。
        """

        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
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


class ECAAttention(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        kernel_size: 一维通道卷积核大小。
    Return:
        无直接返回值，forward 返回 ECA 注意力增强后的特征图。
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Args:
            channels: 输入特征图通道数。
            kernel_size: 一维通道卷积核大小。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            ECA 通道注意力增强后的特征图。
        """

        # 1. 全局池化得到每个通道的响应强度。
        weight = self.avg_pool(x).squeeze(-1).transpose(1, 2)

        # 2. 一维卷积建模相邻通道之间的局部依赖。
        weight = self.conv(weight).transpose(1, 2).unsqueeze(-1)
        weight = self.sigmoid(weight)
        return x * weight


class CoordinateAttention(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        reduction: 坐标注意力中间通道压缩比例。
    Return:
        无直接返回值，forward 返回坐标注意力增强后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 32):
        """
        Args:
            channels: 输入特征图通道数。
            reduction: 坐标注意力中间通道压缩比例。
        Return:
            无直接返回值。
        """

        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.height_gate = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.width_gate = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            坐标注意力增强后的特征图。
        """

        _, _, height, width = x.shape

        # 1. 分别沿宽度和高度聚合，保留位置信息。
        height_context = F.adaptive_avg_pool2d(x, (height, 1))
        width_context = F.adaptive_avg_pool2d(x, (1, width)).transpose(2, 3)

        # 2. 共享变换后再拆分为高度注意力和宽度注意力。
        context = torch.cat([height_context, width_context], dim=2)
        context = self.shared(context)
        height_context, width_context = torch.split(context, [height, width], dim=2)
        width_context = width_context.transpose(2, 3)

        # 3. 坐标方向权重共同增强病灶空间位置表达。
        height_weight = self.sigmoid(self.height_gate(height_context))
        width_weight = self.sigmoid(self.width_gate(width_context))
        return x * height_weight * width_weight


class CBAMAttention(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        reduction: 通道注意力压缩比例。
        kernel_size: 空间注意力卷积核大小。
    Return:
        无直接返回值，forward 返回 CBAM 注意力增强后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: 输入特征图通道数。
            reduction: 通道注意力压缩比例。
            kernel_size: 空间注意力卷积核大小。
        Return:
            无直接返回值。
        """

        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            CBAM 通道和空间注意力增强后的特征图。
        """

        # 1. 通道注意力同时使用平均池化和最大池化。
        avg_weight = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_weight = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        x = x * self.sigmoid(avg_weight + max_weight)

        # 2. 空间注意力突出病灶相关区域。
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial(torch.cat([avg_map, max_map], dim=1))
        return x * spatial_weight


class ResidualAttentionUnit(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        attention: 注意力模块。
    Return:
        无直接返回值，forward 返回残差注意力增强后的特征图。
    """

    def __init__(self, channels: int, attention: nn.Module):
        """
        Args:
            channels: 输入特征图通道数。
            attention: 注意力模块。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.local = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, activation=False),
        )
        self.attention = attention
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            残差注意力增强后的特征图。
        """

        # 1. 局部分支提取该尺度下的纹理特征。
        local_feature = self.local(x)

        # 2. 注意力分支重标定关键通道或空间位置。
        attention_feature = self.attention(local_feature)

        # 3. 残差缩放避免新增注意力破坏预训练骨干表达。
        return x + self.scale * attention_feature


class MultiScaleFusionHead(nn.Module):
    """
    Args:
        shallow_channels: 浅层特征通道数。
        middle_channels: 中层特征通道数。
        deep_channels: 深层特征通道数。
        fusion_dim: 每个尺度投影后的通道维度。
        num_classes: 分类类别数量。
        dropout: Dropout 概率。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(
        self,
        shallow_channels: int,
        middle_channels: int,
        deep_channels: int,
        fusion_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        """
        Args:
            shallow_channels: 浅层特征通道数。
            middle_channels: 中层特征通道数。
            deep_channels: 深层特征通道数。
            fusion_dim: 每个尺度投影后的通道维度。
            num_classes: 分类类别数量。
            dropout: Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.shallow_proj = ConvBNAct(shallow_channels, fusion_dim, kernel_size=1)
        self.middle_proj = ConvBNAct(middle_channels, fusion_dim, kernel_size=1)
        self.deep_proj = ConvBNAct(deep_channels, fusion_dim, kernel_size=1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.SiLU(inplace=True),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, shallow: torch.Tensor, middle: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shallow: 浅层特征图，形状为 [batch_size, shallow_channels, height, width]。
            middle: 中层特征图，形状为 [batch_size, middle_channels, height, width]。
            deep: 深层特征图，形状为 [batch_size, deep_channels, height, width]。
        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        # 1. 将浅层、中层、深层特征统一投影到相同维度。
        shallow = self.shallow_proj(shallow)
        middle = self.middle_proj(middle)
        deep = self.deep_proj(deep)

        # 2. 全局池化得到每个尺度的分类向量。
        shallow_vec = F.adaptive_avg_pool2d(shallow, 1).flatten(1)
        middle_vec = F.adaptive_avg_pool2d(middle, 1).flatten(1)
        deep_vec = F.adaptive_avg_pool2d(deep, 1).flatten(1)

        # 3. 根据样本自适应学习三个尺度的融合权重。
        concat_vec = torch.cat([shallow_vec, middle_vec, deep_vec], dim=1)
        scale_weight = self.fusion_gate(concat_vec)
        shallow_vec = shallow_vec * scale_weight[:, 0:1]
        middle_vec = middle_vec * scale_weight[:, 1:2]
        deep_vec = deep_vec * scale_weight[:, 2:3]

        # 4. 融合多尺度向量后输出分类结果。
        fused = torch.cat([shallow_vec, middle_vec, deep_vec], dim=1)
        return self.classifier(fused)


class MedFuseNet(nn.Module):
    """
    Args:
        num_classes: 皮肤病分类类别数量。
        pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
        fusion_dim: 每个尺度投影后的通道维度。
        dropout: 分类头 Dropout 概率。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(
        self,
        num_classes: int = 23,
        pretrained: bool = True,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_classes: 皮肤病分类类别数量。
            pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
            fusion_dim: 每个尺度投影后的通道维度。
            dropout: 分类头 Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        self.features = backbone.features

        self.shallow_attention = ResidualAttentionUnit(
            channels=32,
            attention=ECAAttention(channels=32),
        )
        self.middle_attention = ResidualAttentionUnit(
            channels=136,
            attention=CoordinateAttention(channels=136),
        )
        self.deep_attention = ResidualAttentionUnit(
            channels=1536,
            attention=CBAMAttention(channels=1536),
        )
        self.head = MultiScaleFusionHead(
            shallow_channels=32,
            middle_channels=136,
            deep_channels=1536,
            fusion_dim=fusion_dim,
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

        # 只初始化新增注意力和融合分类头，避免覆盖 EfficientNet-B3 预训练权重。
        new_layers = [
            self.shallow_attention,
            self.middle_attention,
            self.deep_attention,
            self.head,
        ]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量，形状为 [batch_size, 3, height, width]。
        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        shallow = None
        middle = None

        # 1. 顺序运行 EfficientNet-B3，并取浅层、中层、深层三类特征。
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 2:
                shallow = self.shallow_attention(x)
            elif index == 5:
                middle = self.middle_attention(x)

        # 2. 最后一层输出作为深层语义特征。
        deep = self.deep_attention(x)

        # 3. 多尺度融合后完成分类。
        if shallow is None or middle is None:
            raise RuntimeError("EfficientNet-B3 多尺度特征提取失败。")
        return self.head(shallow, middle, deep)


if __name__ == "__main__":
    model = MedFuseNet(num_classes=23, pretrained=False)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"输出形状: {tuple(output.shape)}")
