"""Classification model evaluation script."""

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
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model.classification_factory import (
    SUPPORTED_CLASSIFICATION_MODELS,
    create_classification_model,
)
from utils.config_handler import model_conf
from utils.config_handler import test_evaluate_conf
from utils.dataset import build_val_transform
from utils.device import device_summary, resolve_device
from utils.metrics import calculate_multiclass_auc
from utils.path_tool import get_abs_path


def get_transforms():
    """
    Args:

    Return:
        Evaluation transform aligned with validation input size.
    """
    return build_val_transform()


def load_checkpoint(model, checkpoint_path, device):
    """
    Args:
        model: 待加载权重的分类模型。
        checkpoint_path: 用户指定的模型权重文件路径。
        device: 权重加载到的计算设备。

    Return:
        已加载权重并切换到评估模式的模型。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 1. 兼容训练脚本保存的完整 checkpoint，也兼容直接保存的 state_dict。
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
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
    """Create a model by name."""
    return create_classification_model(model_name, num_classes=num_classes, pretrained=False)


def resolve_weight_path(weight_path):
    """
    Args:
        weight_path: 用户指定的模型权重路径，支持绝对路径或项目根目录相对路径。

    Return:
        规范化后的绝对权重路径。
    """
    if os.path.isabs(weight_path):
        return weight_path

    return get_abs_path(weight_path)




def main(model_name, weight_path, device_name):
    """
    Args:
        model_name: 需要评估的模型结构名称。
        weight_path: 用户指定的模型权重文件路径。
        device_name: 用户指定的评估设备，如 auto、cpu、cuda:0、mlu:0。

    Return:
        None
    """
    device = resolve_device(device_name)
    num_classes = model_conf["num_classes"]
    print(f"=" * 50)
    print(f"评估模型: {model_name}")
    print(f"类别数: {num_classes}")
    print(f"=" * 50)
    
    model = create_model(model_name, num_classes)
    model = model.to(device)
    print(device_summary(device))
    
    test_dataset = ImageFolder(root=test_evaluate_conf["eval_data"], transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=test_evaluate_conf["batch_size"], shuffle=False)
    print(f"数据集大小: {len(test_dataset)}")
    print(f"类别数: {len(test_dataset.classes)}")
    
    model_path = resolve_weight_path(weight_path)
    if os.path.isfile(model_path):
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
    auc = calculate_multiclass_auc(labels, scores)
    auc_text = f"{auc:.4f}" if auc is not None else "N/A"
    params = sum(p.numel() for p in model.parameters())
    print(f"\n========== 评估结果 ==========")
    print(f"参数量 (parameters):{params}")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"AUC:               {auc_text}")
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
    parser.add_argument('--model', type=str, default='efficientnet_b3',
                        choices=SUPPORTED_CLASSIFICATION_MODELS,
                        help='模型名称')
    parser.add_argument(
        '--weight-path',
        type=str,
        required=True,
        help='模型权重文件路径，如 variables/efficientnet_b3/best_model.pth.tar'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='评估设备: auto / cpu / cuda:0 / 0 / mlu:0'
    )
    args = parser.parse_args()
    
    main(args.model, args.weight_path, args.device)
