import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


#1.Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


#2.SE(Squeeze-and-Excitation)模块
class SE(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        #参2：压缩比例
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            #1.自适应全局平均池化将 H×W 压成 1×1
            nn.AdaptiveAvgPool2d(1),
            #2.降维，减少计算量
            nn.Conv2d(in_channels, se_channels, 1),
            Swish(),
            #3.升维
            nn.Conv2d(se_channels, in_channels, 1),
            #4.Sigmoid激活，得到权重 0-1
            nn.Sigmoid()
        )

    def forward(self, x):
        #乘回原特征（通道注意力）
        return x * self.se(x)



#3.CBAM = 通道注意力 + 空间注意力
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            #1.自适应全局平均池化将 H×W 压成 1×1
            nn.AdaptiveAvgPool2d(1),
            #2.降维，减少计算量
            nn.Conv2d(channels, channels // reduction, 1),
            Swish(),
            #3.升维
            nn.Conv2d(channels // reduction, channels, 1),
            #4.Sigmoid激活，得到权重 0-1
            nn.Sigmoid()
        )
        #空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        #乘回原特征（通道注意力）  [B,C,H,W]
        x = x * ca
        #平均池化 [B,1,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #最大池化 [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #拼接 [B,2,H,W]
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        #乘回原特征（空间注意力）
        return x * sa


# MBConv 模块
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, use_cbam=True):
        super().__init__()
        self.stride = stride
        #使用CBAM
        self.use_cbam = use_cbam
        hidden_dim = in_channels * expand_ratio

        layers = []
        #1.升维：1×1 卷积将通道数扩大 expand_ratio 倍
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, momentum=0.01, eps=1e-3),
                Swish()
            ])

        layers.extend([
            #深度可分离卷积：空间特征提取    groups=hidden_dim 表示每一个通道单独卷积
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=0.01, eps=1e-3),
            Swish(),
            #注意力模块：SE 或 CBAM
            SE(hidden_dim) if not use_cbam else CBAM(hidden_dim),
            #降维：1×1 卷积将通道数恢复到 out_channels
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        ])

        self.conv = nn.Sequential(*layers)
        #残差连接条件：stride=1 且 输入输出通道数相同
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.skip:
            #残差连接
            return x + self.conv(x)
        #无残差连接
        return self.conv(x)


#CustomSkinNet 主模型模块。
class CustomSkinNet(nn.Module):
    def __init__(self, num_classes=23, width_coef=1.5,  pretrained=True):
        #参1：类别数  参2：通道乘数  参3：深度乘数  参4：是否加载预训练权重
        super().__init__()

        # B0 配置: [输入通道, 输出通道, 扩展比, 卷积核, 步长, 重复次数]
        self.config = [
            [32, 16, 1, 3, 1, 1],
            [16, 24, 6, 3, 2, 2],
            [24, 40, 6, 5, 2, 2],
            [40, 80, 6, 3, 2, 3],
            [80, 112, 6, 5, 1, 3],
            [112, 192, 6, 5, 2, 4],
            [192, 320, 6, 3, 1, 1]
        ]

        # 调整通道数
        def _round_channels(channels):
            #32*1.5=48
            new_channels = max(1,int(channels*width_coef))
            return new_channels

        # Stem 主干网络
        self.stem = nn.Sequential(
            #[1,3,224,224] -> [1,48,112,112]
            nn.Conv2d(3, _round_channels(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_round_channels(32), momentum=0.01, eps=1e-3),
            Swish()
        )

        # Blocks 特征提取网络
        blocks = []
        #输入通道  输出通道  扩展比  卷积核大小  步长  重复次数
        for in_ch, out_ch, expand_ratio, kernel, stride, num_repeat in self.config:
            in_ch = _round_channels(in_ch)
            out_ch = _round_channels(out_ch)
            for i in range(num_repeat):
                s = stride if i == 0 else 1  # 第一个 block 用配置的 stride，后续用 1
                blocks.append(MBConv(in_ch, out_ch, expand_ratio, kernel, s))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        """
        block:
        [
        MBConv(48,24,1,3,1)
        
        MBConv(24,36,6,3,2)
        MBConv(36,36,6,3,1)
        
        MBConv(36,60,6,5,2)
        MBConv(60,60,6,5,1)
        
        MBConv(60,120,6,3,2)
        MBConv(120,120,6,3,1)
        MBConv(120,120,6,3,1)
        
        MBConv(120,166,6,5,1)
        MBConv(166,166,6,5,1)
        MBConv(166,166,6,5,1)
        
        MBConv(166,288,6,5,2)
        MBConv(288,288,6,5,1)
        MBConv(288,288,6,5,1)
        MBConv(288,288,6,5,1)
        
        MBConv(288,480,6,3,1)
        ]
        """
        # Head 分类头
        self.head = nn.Sequential(
            nn.Conv2d(_round_channels(320), _round_channels(1280), 1, bias=False),
            nn.BatchNorm2d(_round_channels(1280), momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),    #[B, 1280, 1, 1]
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(_round_channels(1280), num_classes)
        )

        # 加载预训练权重
        if pretrained:
            self._load_pretrained()

        # 初始化权重
        self._init_weights()


    #加载预训练权重
    def _load_pretrained(self):
        try:
            # 加载 EfficientNet-B0 预训练模型
            pretrained_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            pretrained_dict = pretrained_model.state_dict()
            
            model_dict = self.state_dict()
            
            # 只加载形状匹配的权重（名称和维度相同）
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("已加载 EfficientNet-B0 预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")


    #初始化权重
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #Kaiming 初始化（适合 ReLU/Swish 激活函数）
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                #BatchNorm 初始化：权重为 1，偏置为 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                #Linear 初始化：正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = CustomSkinNet(num_classes=23, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输出: {model(x).shape}")
