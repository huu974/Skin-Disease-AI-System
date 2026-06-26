"""
获取分类模型
"""

import os
from torchvision import models
from torch import nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from utils.config_handler import model_conf
class MyModel(nn.Module):
    def __init__(self,model, num_classes):
        super(MyModel, self).__init__()
        self.model = model
        self.num_classes = num_classes



    def model_classifier(self):
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4)
            ,self.model.classifier[-1]
        )
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.num_classes)
        # 初始化最后一层参数
        nn.init.kaiming_normal_(self.model.classifier[-1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.model.classifier[-1].bias)
        return self.model






if __name__ == '__main__':
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', model_conf["save_path"])
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    xiaohui = MyModel(model=model,  num_classes=model_conf["num_classes"])
    xiaohui = xiaohui.model_classifier()
    print(f"最后层输出维度: {xiaohui.classifier[-1].out_features}")
    print("模型加载完成")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(xiaohui)