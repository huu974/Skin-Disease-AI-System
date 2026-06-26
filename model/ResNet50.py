import os
import torch.nn as nn
from torchvision import models, transforms
from utils.config_handler import model_conf




#创建模型
class ResNet50Classifier(nn.Module):
    def __init__(self,num_classes,pretrained=True):
        super(ResNet50Classifier,self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.num_classes = num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        #初始化最后一层权重
        nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_out',nonlinearity='relu')
        nn.init.zeros_(self.model.fc.bias)

    def forward(self,x):
        return self.model(x)




if __name__ == '__main__':
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', model_conf["save_path"])
    model = ResNet50Classifier(num_classes=model_conf["num_classes"],pretrained=True)
    print(model)