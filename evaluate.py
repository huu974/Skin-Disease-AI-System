"""
分类模型评估脚本
支持评估 EfficientNet-B3、ResNet50、CustomSkinNet
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from model.PanDerm import MyModel
from model.ResNet50 import ResNet50Classifier
from model.custom_skin_net import CustomSkinNet
from utils.config_handler import model_conf
from utils.config_handler import test_evaluate_conf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            scores = torch.softmax(outputs, dim=1)
            preds = torch.argmax(scores, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, annot_kws={"size": 8}, cmap='Blues')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")


def create_model(model_name, num_classes):
    """根据模型名称创建模型"""
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), model_conf["save_path"])
    
    if model_name == 'efficientnet_b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        xiaohui = MyModel(model=model, num_classes=num_classes)
        return xiaohui.model_classifier()
    
    elif model_name == 'resnet50':
        return ResNet50Classifier(num_classes=num_classes, pretrained=True)
    
    elif model_name == 'custom_skin_net':
        return CustomSkinNet(
            num_classes=num_classes,
            width_coef=1.5,
            depth_coef=1.4,
            pretrained=False
        )
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def get_model_path(model_name):
    """根据模型名称获取模型权重路径"""
    base_path = './variables'
    model_dir = os.path.join(base_path, model_name)
    best_model = os.path.join(model_dir, 'best_model.pth.tar')
    
    if os.path.exists(best_model):
        return best_model
    
    fallback = os.path.join(base_path, 'best_model.pth.tar')
    if os.path.exists(fallback):
        return fallback
    
    return best_model




def main(model_name):
    num_classes = model_conf["num_classes"]
    print(f"=" * 50)
    print(f"评估模型: {model_name}")
    print(f"类别数: {num_classes}")
    print(f"=" * 50)
    
    model = create_model(model_name, num_classes)
    model = model.to(device)
    print(f"设备: {device}")
    
    test_dataset = ImageFolder(root=test_evaluate_conf["eval_data"], transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=test_evaluate_conf["batch_size"], shuffle=False)
    print(f"数据集大小: {len(test_dataset)}")
    print(f"类别数: {len(test_dataset.classes)}")
    
    model_path = get_model_path(model_name)
    if os.path.exists(model_path):
        model = load_checkpoint(model, model_path, device)
        print(f"已加载模型权重: {model_path}")
    else:
        print(f"未找到模型权重文件: {model_path}")
        return
    
    print("正在评估模型...")
    labels, preds, scores = evaluate(model, test_loader, device)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro')
    
    print(f"\n========== 评估结果 ==========")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"=" * 50)
    
    cm = confusion_matrix(labels, preds)
    save_path = f'confusion_matrix_{model_name}.png'
    plot_confusion_matrix(cm, test_dataset.classes, save_path)
    
    print('\n========== 分类报告 ==========')
    print(classification_report(
        labels, preds, target_names=test_dataset.classes, zero_division=0
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='皮肤病分类模型评估')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['efficientnet_b3', 'resnet50', 'custom_skin_net'],
                        help='模型名称')
    args = parser.parse_args()
    
    main(args.model)
