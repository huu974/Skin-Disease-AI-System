"""EfficientNet-B3 + 轻量 Transformer 分类头。"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class ConvBNAct(nn.Module):
    """
    Args:
        in_channels: 输入特征图通道数。
        out_channels: 输出特征图通道数。
        kernel_size: 卷积核大小。
        groups: 分组卷积的组数。
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
            groups: 分组卷积的组数。
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
            经过 trunk-mask 残差注意力增强后的特征图。
        """

        # 1. trunk 分支提取可被注意力增强的局部语义特征。
        trunk_feature = self.trunk(x)

        # 2. mask 分支生成通道注意力权重。
        mask_weight = self.mask(x)

        # 3. 使用 1 + mask 的残差注意力形式，避免破坏 EfficientNet 原始特征。
        attention_feature = trunk_feature * (1.0 + mask_weight)
        return x + self.scale * attention_feature


class TransformerBlock(nn.Module):
    """
    Args:
        dim: token 特征维度。
        heads: 多头注意力头数。
        mlp_dim: 前馈网络隐藏层维度。
        dropout: Dropout 概率。
    Return:
        无直接返回值，forward 返回 Transformer 编码后的 token 序列。
    """

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        """
        Args:
            dim: token 特征维度。
            heads: 多头注意力头数。
            mlp_dim: 前馈网络隐藏层维度。
            dropout: Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token 序列，形状为 [batch_size, token_count, dim]。
        Return:
            Transformer 编码后的 token 序列。
        """

        # 1. 自注意力用于建模皮损不同区域之间的全局关系。
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output

        # 2. 前馈网络用于增强每个 token 的非线性表达。
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LightTransformerHead(nn.Module):
    """
    Args:
        in_channels: EfficientNet-B3 输出特征图通道数。
        num_classes: 分类类别数量。
        dim: Transformer token 特征维度。
        depth: TransformerBlock 堆叠层数。
        heads: 多头注意力头数。
        mlp_dim: 前馈网络隐藏层维度。
        token_grid: 自适应池化后的 token 网格大小。
        dropout: Dropout 概率。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 512,
        token_grid: tuple[int, int] = (2, 4),
        dropout: float = 0.2,
    ):
        """
        Args:
            in_channels: EfficientNet-B3 输出特征图通道数。
            num_classes: 分类类别数量。
            dim: Transformer token 特征维度。
            depth: TransformerBlock 堆叠层数。
            heads: 多头注意力头数。
            mlp_dim: 前馈网络隐藏层维度。
            token_grid: 自适应池化后的 token 网格大小。
            dropout: Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        token_count = token_grid[0] * token_grid[1]
        self.token_pool = nn.AdaptiveAvgPool2d(token_grid)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_count + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EfficientNet-B3 输出特征图，形状为 [batch_size, in_channels, height, width]。
        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        # 1. 将 CNN 特征压缩成固定数量的空间 token。
        x = self.token_pool(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        # 2. 加入 cls token 和位置编码，保留全局分类入口。
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        # 3. 使用轻量 Transformer 建模皮损区域间关系。
        for block in self.blocks:
            x = block(x)

        # 4. 使用 cls token 输出最终分类结果。
        cls_feature = self.norm(x[:, 0])
        return self.classifier(cls_feature)


class EffVitNet(nn.Module):
    """
    Args:
        num_classes: 皮肤病分类类别数量。
        pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
        dim: Transformer token 特征维度。
        depth: TransformerBlock 堆叠层数。
        heads: 多头注意力头数。
        mlp_dim: 前馈网络隐藏层维度。
        token_grid: 自适应池化后的 token 网格大小。
        dropout: Dropout 概率。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(
        self,
        num_classes: int = 23,
        pretrained: bool = True,
        dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 512,
        token_grid: tuple[int, int] = (2, 4),
        dropout: float = 0.2,
    ):
        """
        Args:
            num_classes: 皮肤病分类类别数量。
            pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
            dim: Transformer token 特征维度。
            depth: TransformerBlock 堆叠层数。
            heads: 多头注意力头数。
            mlp_dim: 前馈网络隐藏层维度。
            token_grid: 自适应池化后的 token 网格大小。
            dropout: Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        feature_dim = 1536

        self.features = backbone.features
        self.attention = ResidualAttentionBlock(channels=feature_dim)
        self.head = LightTransformerHead(
            in_channels=feature_dim,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            token_grid=token_grid,
            dropout=dropout,
        )
        self._init_head()

    def _init_head(self) -> None:
        """
        Args:
            无。
        Return:
            无直接返回值。
        """

        # 只初始化新增残差注意力和 Transformer Head，避免覆盖 EfficientNet-B3 预训练权重。
        for layer in self.attention.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.trunc_normal_(self.head.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.pos_embedding, std=0.02)
        for layer in self.head.modules():
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

        # 1. 使用 EfficientNet-B3 提取局部纹理和形态特征。
        x = self.features(x)

        # 2. 使用残差注意力增强病灶相关区域特征。
        x = self.attention(x)

        # 3. 使用轻量 Transformer Head 建模全局区域关系并分类。
        return self.head(x)


if __name__ == "__main__":
    model = EffVitNet(num_classes=23, pretrained=False)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"输出形状: {tuple(output.shape)}")
