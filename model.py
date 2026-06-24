import torch
import torch.nn as nn
from scipy.cluster.hierarchy import weighted
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class Swish(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)



class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        #通道注意力
        self.channel_attention = nn.Sequential(
            #全局平均池化，获取该通道的整体强度
            nn.AdaptiveAvgPool2d(1),
            #降维，减少计算量，防止过拟合，同时融合不同通道的特征
            nn.Conv2d(channels, channels // reduction, 1),
            #非线性激活，加入非线性，让网络学习不同的通道的特征
            Swish(),
            #升维，出去冗余信息，保留重要信息，使通道数与原始特征图一致，方便后面做乘法
            nn.Conv2d(channels // reduction, channels, 1),
            #非线性激活，将输出压缩到0-1之间，变成注意力权重，越接近1，则该通道越重要，特征越强，越接近0，则该通道不重要，特征抑制
            nn.Sigmoid()
        )
        #空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )


    def forward(self, x):
        #通道注意力
        ca = self.channel_attention(x)
        #乘回原特征
        x = x * ca
        #平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #拼接
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        #乘回原特征（空间注意力）
        return x * sa





class CustomSkinNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()

        #stem主干
        self.stem = nn.Sequential(
            nn.Conv2d(3,48,3,2,1,bias=False),
            nn.BatchNorm2d(48,momentum=0.01,eps=1e-3),
            Swish()
        )


        #MBConv(48,24,1,3,1)  in_c,out_c,expand_ratio,kernel_size,stride
        #blocks特征提取网络
        self.block1 = nn.Sequential(
            #深度可分离卷积：空间特征提取
            nn.Conv2d(48,48,3,1,groups=48,bias=False),
            nn.BatchNorm2d(48,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(48),
            #降维，将通道数恢复到输出通道
            nn.Conv2d(48,24,1,bias=False),
            nn.BatchNorm2d(24,momentum=0.01,eps=1e-3)
        )


        #MBConv(24,36,6,3,2)
        self.block2 = nn.Sequential(
            nn.Conv2d(24,144,1,bias=False),
            nn.BatchNorm2d(144,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(144,144,3,2,groups=144,bias=False),
            nn.BatchNorm2d(144,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(144),
            nn.Conv2d(144,36,1,bias=False),
            nn.BatchNorm2d(36,momentum=0.01,eps=1e-3),
        )




        #MBConv(36,36,6,3,1)
        self.block3 = nn.Sequential(
            nn.Conv2d(36,216,1,bias=False),
            nn.BatchNorm2d(216,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(216,216,3,1,groups=216,bias=False,padding=1),
            nn.BatchNorm2d(216,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(216),
            nn.Conv2d(216,36,1,bias=False),
            nn.BatchNorm2d(36,momentum=0.01,eps=1e-3)
        )

        self.skip3=True




        #MBConv(36,60,6,5,2)
        self.block4 = nn.Sequential(
            nn.Conv2d(36,216,1,bias=False),
            nn.BatchNorm2d(216,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(216,216,5,2,groups=216,bias=False,padding=1),
            nn.BatchNorm2d(216,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(216),
            nn.Conv2d(216,60,1,bias=False),
            nn.BatchNorm2d(60,momentum=0.01,eps=1e-3)
        )



        #MBConv(36,60,6,5,1)
        self.block5 = nn.Sequential(
            nn.Conv2d(60,216,1,bias=False),
            nn.BatchNorm2d(216,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(216,216,5,1,groups=216,bias=False,padding=2),
            nn.BatchNorm2d(216,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(216),
            nn.Conv2d(216,60,1,bias=False),
            nn.BatchNorm2d(60,momentum=0.01,eps=1e-3)
        )




        #MBConv(60,120,6,3,2)
        self.block6 = nn.Sequential(
            nn.Conv2d(60,360,1,bias=False),
            nn.BatchNorm2d(360,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(360,360,3,2,groups=360,bias=False,padding=1),
            nn.BatchNorm2d(360,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(360),
            nn.Conv2d(360,120,1,bias=False),
            nn.BatchNorm2d(120,momentum=0.01,eps=1e-3)
        )


        #MBConv(120,120,6,3,1)
        self.block7 = nn.Sequential(
            nn.Conv2d(120,720,1,bias=False),
            nn.BatchNorm2d(720,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(720,720,3,1,groups=720,bias=False,padding=1),
            nn.BatchNorm2d(720,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(720),
            nn.Conv2d(720,120,1,bias=False),
            nn.BatchNorm2d(120,momentum=0.01,eps=1e-3)
        )

        self.skip7 = True



        #MBConv(120,120,6,3,1)
        self.block8 = nn.Sequential(
            nn.Conv2d(120,720,1,bias=False),
            nn.BatchNorm2d(720,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(720,720,3,1,groups=720,bias=False,padding=1),
            nn.BatchNorm2d(720,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(720),
            nn.Conv2d(720,120,1,bias=False),
            nn.BatchNorm2d(120,momentum=0.01,eps=1e-3)
        )

        self.skip8 = True


        #MBConv(120,166,6,5,1)
        self.block9 = nn.Sequential(
            nn.Conv2d(120,720,1,bias=False),
            nn.BatchNorm2d(720,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(720,720,5,1,groups=720,bias=False,padding=2),
            nn.BatchNorm2d(720,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(720),
            nn.Conv2d(720,166,1,bias=False),
            nn.BatchNorm2d(166,momentum=0.01,eps=1e-3)
        )




        #MBConv(166,166,6,5,1)
        self.block10 = nn.Sequential(
            nn.Conv2d(166,966,1,bias=False),
            nn.BatchNorm2d(966,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(966,966,5,1,groups=966,bias=False,padding=2),
            nn.BatchNorm2d(966,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(966),
            nn.Conv2d(966,166,1,bias=False),
            nn.BatchNorm2d(166,momentum=0.01,eps=1e-3)
        )

        self.skip10 = True




        #MBConv(166,166,6,5,1)
        self.block11 = nn.Sequential(
            nn.Conv2d(166,966,1,bias=False),
            nn.BatchNorm2d(966,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(966,966,5,1,groups=966,bias=False,padding=2),
            nn.BatchNorm2d(966,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(966),
            nn.Conv2d(966,166,1,bias=False),
            nn.BatchNorm2d(166,momentum=0.01,eps=1e-3)
        )

        self.skip11 = True






        #MBConv(166,288,6,5,2)
        self.block12 = nn.Sequential(
            nn.Conv2d(166,966,1,bias=False),
            nn.BatchNorm2d(966,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(966,966,5,2,groups=966,bias=False,padding=1),
            nn.BatchNorm2d(966,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(966),
            nn.Conv2d(966,288,1,bias=False),
            nn.BatchNorm2d(288,momentum=0.01,eps=1e-3)
        )







        #MBConv(288,288,6,5,1)
        self.block13 = nn.Sequential(
            nn.Conv2d(288,1728,1,bias=False),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(1728,1728,5,1,groups=1728,bias=False,padding=2),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(1728),
            nn.Conv2d(1728,288,1,bias=False),
            nn.BatchNorm2d(288,momentum=0.01,eps=1e-3)
        )


        self.skip13=True





        #MBConv(288,288,6,5,1)
        self.block14 = nn.Sequential(
            nn.Conv2d(288,1728,1,bias=False),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(1728,1728,5,1,groups=1728,bias=False,padding=2),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(1728),
            nn.Conv2d(1728,288,1,bias=False),
            nn.BatchNorm2d(288,momentum=0.01,eps=1e-3)
        )


        self.skip14=True







        #MBConv(288,288,6,5,1)
        self.block15 = nn.Sequential(
            nn.Conv2d(288,1728,1,bias=False),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(1728,1728,5,1,groups=1728,bias=False,padding=2),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(1728),
            nn.Conv2d(1728,288,1,bias=False),
            nn.BatchNorm2d(288,momentum=0.01,eps=1e-3)
        )


        self.skip15=True






        #MBConv(288,480,6,3,1)
        self.block16 = nn.Sequential(
            nn.Conv2d(288,1728,1,bias=False),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            nn.Conv2d(1728,1728,3,1,groups=1728,bias=False,padding=1),
            nn.BatchNorm2d(1728,momentum=0.01,eps=1e-3),
            Swish(),
            CBAM(1728),
            nn.Conv2d(1728,480,1,bias=False),
            nn.BatchNorm2d(480,momentum=0.01,eps=1e-3)
        )



        #Head分类头
        self.head = nn.Sequential(
            #1.扩展通道数，将在低维空间提取的特征映射到高维，增强特征的可分性
            nn.Conv2d(480,1920,1,bias=True),
            #2.加速收敛，稳定梯度，减少过拟合
            nn.BatchNorm2d(1920,momentum=0.01,eps=1e-3),
            #3.增加模型非线性能力，提升特征利用率
            Swish(),
            #4.获得每一个通道的重要性数值
            nn.AdaptiveAvgPool2d(1),
            #5.展平，构建线性层的输入
            nn.Flatten(),
            #6.随机失活，抑制过拟合
            nn.Dropout(0.2),
            #7.输出目标类别的特征维度映射
            nn.Linear(1920,3)
        )


        #加载预训练权重
        if pretrained:
            self._load_pretrained()


    def _load_pretrained(self):
        try:
            #使用EfficientNet-B0
            pretrained_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            pretrained_dict = pretrained_model.state_dict()

            model_dict = self.state_dict()

            #只加载形状匹配的权重(名称和维度相同)
            pretrained_dict = { k:v for k ,v in pretrained_dict.items()
                                if k in model_dict and model_dict[k].shape == v.shape
            }

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


            print("已加载 EfficientNet-B0 预训练权重")

        except Exception as e:
            print(f"加载预训练权重失败: {e}")


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                #Kaiming初始化
                nn.init.kaiming_normal_(m.weight,mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)



    def forward(self,x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        #残差连接
        x = x + self.block3(x)
        x = self.block4(x)
        x = x + self.block5(x)
        x = self.block6(x)
        x = x + self.block7(x)
        x = x + self.block8(x)
        x = self.block9(x)
        x = x + self.block10(x)
        x = x + self.block11(x)
        x = self.block12(x)
        x = x + self.block13(x)
        x = x + self.block14(x)
        x = x + self.block15(x)
        x = self.block16(x)
        x = self.head(x)

        return x




if __name__ == "__main__":
    model = CustomSkinNet()
    x = torch.randn(1, 3, 224, 224)
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输出: {model(x).shape}")


