"""
模型测试脚本
"""

import os
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from model.PanDerm import MyModel
from utils.config_handler import model_conf
from utils.config_handler import test_evaluate_conf


# SKIN_DISEASE_CLASSES= [
# "痤疮和酒渣鼻","光化性角化病和基底细胞癌","特应性皮炎",
# "大疱性疾病","蜂窝组织炎和细菌感染","湿疹",
# "发疹和药物性皮炎","脱发","疱疹/HPV",
# "色素性疾病","红斑狼疮","黑色素瘤和痣",
# "甲真菌病","毒葛皮炎","银屑病和扁平苔藓",
# "疥疮和莱姆病","脂溢性角化病和良性肿瘤","系统性疾病",
# "真菌感染","荨麻疹","血管瘤",
# "血管炎","疣和传染性软疣"
# ]

def get_random_test_image():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin diseases', 'test')
    class_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    if not class_folders:
        raise FileNotFoundError("测试集目录为空")
    
    selected_class = random.choice(class_folders)
    class_path = os.path.join(test_dir, selected_class)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise FileNotFoundError(f"类别 {selected_class} 中没有图片")
    
    selected_image = random.choice(images)
    image_path = os.path.join(class_path, selected_image)
    return image_path, selected_class





transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



#1.测试分类模型
def test_classifier(image_path,model_name='custom_skin_net'):
    print('开始测试分类模型...')
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 根据模型名称加载分类模型
    if model_name == 'resnet50':
        from model.ResNet50 import ResNet50Classifier
        model = ResNet50Classifier(num_classes=model_conf["num_classes"], pretrained=False)
    elif model_name == 'efficientnet_b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        xiaohui = MyModel(model=model, num_classes=model_conf["num_classes"])
        model = xiaohui.model_classifier().to(device)
        checkpoint = torch.load(test_evaluate_conf['classification_model'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == 'custom_skin_net':
        from model.custom_skin_net import CustomSkinNet
        model = CustomSkinNet(
            num_classes=model_conf["num_classes"],
            width_coef=1.5,
            depth_coef=1.4,
            pretrained=False
        ).to(device)
        checkpoint = torch.load("variables/custom_skin_net/best_model.pth.tar", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"不支持的模型: {model_name}")


    model = model.to(device)
    print(f'使用模型：{model_name}')
    #测试开始
    model.eval()

    #图片预处理
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)


    #推理
    with torch.no_grad():
        outputs = model(image_tensor)
        proabilities = torch.softmax(outputs, dim=1)
        confidence,pred = torch.max(proabilities, 1)

    diease = test_evaluate_conf['class_names'][pred.item()]

    #打印测试结果
    print(f'分类结果：{diease}，置信度：{confidence.item():.2%}')


    return diease, confidence







if __name__ == '__main__':
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print(f'使用设备：{device}')

     # test_image,_ = get_random_yolo_image()
     # test_image = "skin diseases/test/Light Diseases and Disorders of Pigmentation/sun-damaged-skin-27.jpg"
     test_image,true_class = get_random_test_image()
     print(f'测试图片路径：{test_image}')
     print()



     models = ['efficientnet_b3', 'custom_skin_net']
     results = {}

     for model_name in models:
         print(f'\n>>> {model_name}:')
         disease, confidence = test_classifier(test_image, model_name=model_name)
         results[model_name] = {'disease': disease, 'confidence': confidence}

     print('\n' + '='*50)
     print('测试完成！')

