"""EfficientNet-B3 + 残差注意力 + CfC/AutoNCP 液态神经网络分类头。"""
import os

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from utils.config_handler import model_conf

try:
    from ncps.torch import CfC
    from ncps.wirings import AutoNCP
except ImportError:
    CfC = None
    AutoNCP = None


class ResidualAttention(nn.Module):
    """
    用于卷积特征图的残差通道-空间注意力模块。

    Args:
        channels: 输入特征图的通道数。
        reduction: 通道注意力中的通道压缩比例。

    Return:
        无直接返回值，forward 方法返回增强后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入特征图的通道数。
            reduction: 通道注意力中的通道压缩比例。

        Return:
            无直接返回值。
        """

        super().__init__()
        reduced_channels = max(1, channels // reduction)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]。

        Return:
            加入残差注意力后的特征图。
        """

        # 1. 通道注意力用于突出更重要的语义通道。
        channel_weight = self.channel_attention(x)
        channel_feature = x * channel_weight

        # 2. 空间注意力用于突出更重要的图像区域。
        avg_feature = torch.mean(channel_feature, dim=1, keepdim=True)
        max_feature, _ = torch.max(channel_feature, dim=1, keepdim=True)
        spatial_weight = self.spatial_attention(torch.cat([avg_feature, max_feature], dim=1))
        attention_feature = channel_feature * spatial_weight

        # 3. 残差连接用于保留 EfficientNet 原始特征表达，避免注意力过度破坏特征。
        return x + self.residual_scale * attention_feature


class CfCAutoNCPHead(nn.Module):
    """
    CfC/AutoNCP 液态神经网络分类头。

    Args:
        feature_dim: EfficientNet-B3 展平后的特征维度。
        sequence_length: 将特征向量切分成序列后的时间步数量。
        hidden_size: AutoNCP 连接结构中的神经元数量。
        num_classes: 输出类别数量。
        dropout: 送入液态神经网络分类头之前的 Dropout 概率。

    Return:
        无直接返回值，forward 方法返回分类 logits。
    """

    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        """
        Args:
            feature_dim: EfficientNet-B3 展平后的特征维度。
            sequence_length: 将特征向量切分成序列后的时间步数量。
            hidden_size: AutoNCP 连接结构中的神经元数量。
            num_classes: 输出类别数量。
            dropout: 送入液态神经网络分类头之前的 Dropout 概率。

        Return:
            无直接返回值。
        """

        super().__init__()
        if CfC is None or AutoNCP is None:
            raise ImportError("缺少依赖：请先执行 `pip install ncps` 安装 ncps。")
        if feature_dim % sequence_length != 0:
            raise ValueError("feature_dim 必须能够被 sequence_length 整除。")
        if hidden_size <= num_classes:
            raise ValueError("使用 AutoNCP 时，hidden_size 必须大于 num_classes。")

        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.input_size = feature_dim // sequence_length
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        wiring = AutoNCP(hidden_size, num_classes)
        self.rnn = self._create_cfc(self.input_size, wiring)

    def _create_cfc(self, input_size: int, wiring):
        """
        Args:
            input_size: 每个序列时间步的特征维度。
            wiring: AutoNCP 连接结构对象。

        Return:
            CfC 循环神经网络模块。
        """

        # 1. 新版本 ncps 支持显式设置 batch_first 和 return_sequences。
        try:
            return CfC(input_size, wiring, batch_first=True, return_sequences=True)
        except TypeError:
            return CfC(input_size, wiring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 展平后的特征张量，形状为 [batch_size, feature_dim]。

        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        x = self.dropout(x)

        # 1. 将单张图像的特征向量转换成短序列特征。
        x = x.view(x.size(0), self.sequence_length, self.input_size)

        # 2. 为当前 batch 初始化液态神经网络隐藏状态。
        h0 = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        output, _ = self.rnn(x, h0)

        # 3. 使用最后一个时间步的输出进行图像分类。
        if output.dim() == 3:
            output = output[:, -1, :]
        return output


class EfficientNetB3ResidualAttentionLNN(nn.Module):
    """
    EfficientNet-B3 + 残差注意力 + CfC/AutoNCP 液态神经网络分类模型。

    Args:
        num_classes: 疾病分类类别数量。
        pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
        sequence_length: 送入液态神经网络分类头的序列时间步数量。
        hidden_size: AutoNCP 连接结构中的神经元数量。
        attention_reduction: 残差注意力模块中的通道压缩比例。
        dropout: 送入液态神经网络分类头之前的 Dropout 概率。

    Return:
        无直接返回值，forward 方法返回分类 logits。
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        sequence_length: int = 8,
        hidden_size: int = 64,
        attention_reduction: int = 16,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_classes: 疾病分类类别数量。
            pretrained: 是否加载 ImageNet 预训练的 EfficientNet-B3 权重。
            sequence_length: 送入液态神经网络分类头的序列时间步数量。
            hidden_size: AutoNCP 连接结构中的神经元数量。
            attention_reduction: 残差注意力模块中的通道压缩比例。
            dropout: 送入液态神经网络分类头之前的 Dropout 概率。

        Return:
            无直接返回值。
        """

        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(start_dim=1)

        # EfficientNet-B3 最后一层卷积特征通道数为 1536。
        feature_dim = 1536
        self.attention = ResidualAttention(channels=feature_dim, reduction=attention_reduction)
        self.lnn_head = CfCAutoNCPHead(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量，形状为 [batch_size, 3, height, width]。

        Return:
            分类 logits，形状为 [batch_size, num_classes]。
        """

        # 1. 使用 EfficientNet-B3 提取卷积图像特征。
        x = self.features(x)

        # 2. 使用残差注意力增强最后一层特征图。
        x = self.attention(x)

        # 3. 池化特征，并使用 CfC/AutoNCP 液态分类头输出结果。
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.lnn_head(x)
        return x


if __name__ == "__main__":
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', model_conf["save_path"])
    model = EfficientNetB3ResidualAttentionLNN(23)
    print(model)
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")