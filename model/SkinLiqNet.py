"""EfficientNet-B3 + 残差注意力 + 噪声抑制 + 液态分类头。"""

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
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        """
        Args:
            in_channels: 输入特征图通道数。
            out_channels: 输出特征图通道数。
            kernel_size: 卷积核大小。
            stride: 卷积步长。
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


class ResidualAttentionBlock(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        reduction: 注意力掩码分支的通道压缩比例。
        trunk_ratio: 主干分支的中间通道压缩比例。
    Return:
        无直接返回值，forward 返回残差注意力增强后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 16, trunk_ratio: int = 4):
        """
        Args:
            channels: 输入特征图通道数。
            reduction: 注意力掩码分支的通道压缩比例。
            trunk_ratio: 主干分支的中间通道压缩比例。
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

        # 1. trunk 分支保留主干语义特征。
        trunk_feature = self.trunk(x)

        # 2. mask 分支生成 0 到 1 的通道级注意力权重。
        mask_weight = self.mask(x)

        # 3. 使用 1 + mask 的残差注意力形式，避免强行压低原始特征。
        attention_feature = trunk_feature * (1.0 + mask_weight)
        return x + self.scale * attention_feature


class NoiseSuppressor(nn.Module):
    """
    Args:
        channels: 输入特征图通道数。
        reduction: 噪声门控分支的通道压缩比例。
        groups: 逐点分组卷积的组数，用于控制参数量。
    Return:
        无直接返回值，forward 返回噪声抑制后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 16, groups: int = 16):
        """
        Args:
            channels: 输入特征图通道数。
            reduction: 噪声门控分支的通道压缩比例。
            groups: 逐点分组卷积的组数，用于控制参数量。
        Return:
            无直接返回值。
        """

        super().__init__()
        gate_channels = max(16, channels // reduction)
        pointwise_groups = groups if channels % groups == 0 else 1

        self.noise_estimator = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, groups=pointwise_groups, activation=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, gate_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.noise_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。
        Return:
            经过局部高频噪声抑制后的特征图。
        """

        # 1. 平均池化得到低频结构，高频残差更容易包含毛发、反光和采集噪声。
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq

        # 2. 用高频残差估计噪声分量。
        noise = self.noise_estimator(high_freq)

        # 3. 同时使用低频均值和高频强度生成通道噪声门控。
        low_stat = F.adaptive_avg_pool2d(low_freq, output_size=1)
        high_stat = F.adaptive_avg_pool2d(high_freq.abs(), output_size=1)
        noise_gate = self.gate(torch.cat([low_stat, high_stat], dim=1))

        # 4. 以残差方式减去受门控约束的噪声，保留病灶主体特征。
        return x - self.noise_scale * noise_gate * noise


class LiquidNeuronCell(nn.Module):
    """
    Args:
        input_size: 单个序列步输入特征维度。
        hidden_size: 液态隐藏状态维度。
    Return:
        无直接返回值，forward 返回更新后的隐藏状态。
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: 单个序列步输入特征维度。
            hidden_size: 液态隐藏状态维度。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_gate = nn.Linear(input_size, hidden_size)
        self.hidden_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, h_state: torch.Tensor, delta_t: float = 0.1) -> torch.Tensor:
        """
        Args:
            x: 当前序列步输入，形状为 [batch_size, input_size]。
            h_state: 上一时刻隐藏状态，形状为 [batch_size, hidden_size]。
            delta_t: 离散更新步长。
        Return:
            更新后的液态隐藏状态。
        """

        # 1. 候选状态描述当前输入和历史状态的融合结果。
        candidate = torch.tanh(self.input_proj(x) + self.hidden_proj(h_state))

        # 2. 门控项控制每个隐藏单元的更新强度。
        gate = torch.sigmoid(self.input_gate(x) + self.hidden_gate(h_state))

        # 3. 正时间常数用于模拟连续时间状态衰减。
        tau = F.softplus(self.tau).unsqueeze(0) + 1e-4
        dh = (-h_state + candidate) / tau
        return h_state + delta_t * gate * dh


class LiquidHead(nn.Module):
    """
    Args:
        input_size: 单个空间序列步的特征维度。
        hidden_size: 液态隐藏状态维度。
        num_classes: 分类类别数量。
        dropout: 分类前 Dropout 概率。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2):
        """
        Args:
            input_size: 单个空间序列步的特征维度。
            hidden_size: 液态隐藏状态维度。
            num_classes: 分类类别数量。
            dropout: 分类前 Dropout 概率。
        Return:
            无直接返回值。
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiquidNeuronCell(input_size=input_size, hidden_size=hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 空间序列特征，形状为 [batch_size, sequence_length, input_size]。
        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        batch_size = x.size(0)
        h_state = x.new_zeros(batch_size, self.hidden_size)

        # 1. 逐个空间 token 更新液态隐藏状态。
        for step in range(x.size(1)):
            h_state = self.cell(x[:, step, :], h_state)

        # 2. 使用最终隐藏状态进行分类。
        h_state = self.norm(h_state)
        h_state = self.dropout(h_state)
        return self.classifier(h_state)


class SkinLiqNet(nn.Module):
    """
    Args:
        num_classes: 皮肤病分类类别数量。
        pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
        hidden_size: 液态分类头隐藏状态维度。
        dropout: 分类前 Dropout 概率。
        token_grid: 自适应池化后的空间 token 网格大小。
    Return:
        无直接返回值，forward 返回分类 logits。
    """

    def __init__(
        self,
        num_classes: int = 23,
        pretrained: bool = True,
        hidden_size: int = 128,
        dropout: float = 0.2,
        token_grid: tuple[int, int] = (2, 4),
    ):
        """
        Args:
            num_classes: 皮肤病分类类别数量。
            pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
            hidden_size: 液态分类头隐藏状态维度。
            dropout: 分类前 Dropout 概率。
            token_grid: 自适应池化后的空间 token 网格大小。
        Return:
            无直接返回值。
        """

        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        feature_dim = 1536

        self.features = backbone.features
        self.attention = ResidualAttentionBlock(channels=feature_dim)
        self.noise_suppressor = NoiseSuppressor(channels=feature_dim)
        self.token_pool = nn.AdaptiveAvgPool2d(token_grid)
        self.head = LiquidHead(
            input_size=feature_dim,
            hidden_size=hidden_size,
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

        # 只初始化新增模块，避免覆盖 EfficientNet-B3 预训练权重。
        new_layers = [self.attention, self.noise_suppressor, self.head]
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
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量，形状为 [batch_size, 3, height, width]。
        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        # 1. 使用 EfficientNet-B3 提取卷积特征。
        x = self.features(x)

        # 2. 残差注意力增强病灶相关通道响应。
        x = self.attention(x)

        # 3. 噪声抑制模块削弱局部高频干扰。
        x = self.noise_suppressor(x)

        # 4. 将空间特征压缩为短序列，送入液态分类头。
        x = self.token_pool(x)
        x = x.flatten(2).transpose(1, 2)
        return self.head(x)


if __name__ == "__main__":
    model = SkinLiqNet(num_classes=23, pretrained=False)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"输出形状: {tuple(output.shape)}")
