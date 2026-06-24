"""
分类模块主函数
"""

import os
import sys
import signal
import torch
from torch import autocast, nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from utils.config_handler import model_conf
from utils.dataset import get_train_dataloader
from utils.dataset import get_val_dataloader
from utils.arguments import parse
import torch.backends.cudnn as cudnn
from utils.optimizer_Adam import CustomAdam
from model.PanDerm import MyModel
from train_validation import tra_val
from utils.outputwriter import OutputSave
from utils.writer import init_writer
from model.ResNet50 import ResNet50Classifier
from model.custom_skin_net import CustomSkinNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class InterruptHandler:
    def __init__(self, saver):
        self.saver = saver
        self.current_epoch = 0
        
    def handler(self, signum, frame):
        print("\n检测到 Ctrl+C，正在保存模型...")
        self.saver.save_checkpoint(self.current_epoch)
        print(f"模型已保存到 {self.saver.args.save_path}")
        print("下次运行时使用 --resume 参数恢复训练")
        sys.exit(0)


def main():
    args = parse()
    
    # 初始化日志和TensorBoard
    writer = init_writer(args)

    # 设置加速
    cudnn.benchmark = True

    # 获取数据加载器
    train_dataloader = get_train_dataloader(args)
    val_dataloader = get_val_dataloader(args)

    model_name = getattr(args,'model','efficientnet_b3')
    # 获取模型统一预训练模型目录
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), model_conf["save_path"])
    
    # 根据模型名创建子目录保存模型
    args.save_path = os.path.join(args.save_path, model_name)
    
    if model_name == 'resnet50':
        model = ResNet50Classifier(num_classes=model_conf["num_classes"],pretrained=True)


    elif model_name == 'efficientnet_b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        xiaohui = MyModel(model=model,  num_classes=model_conf["num_classes"])
        model = xiaohui.model_classifier()

    elif model_name == 'custom_skin_net':
        model = CustomSkinNet(
            num_classes=model_conf["num_classes"],
            width_coef=1.5,
            depth_coef=1.4,
            pretrained=False
        )

    else:
        raise ValueError(f"不支持的模型: {model_name}")


    model = model.to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = CustomAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 模型保存
    saver = OutputSave(model, args, optimizer)

    # 恢复训练
    start_epoch = 0
    print(f"resume 参数: {args.resume}")  # 加这行
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从第 {start_epoch} 轮恢复训练")

    # 设置 Ctrl+C 捕获
    handler = InterruptHandler(saver)
    signal.signal(signal.SIGINT, handler.handler)

    # 获取训练器与验证器
    traval = tra_val(model, criterion, optimizer, scaler, args, train_dataloader, None, writer)
    val = tra_val(model, criterion, optimizer, scaler, args, None, val_dataloader, writer)

    # 训练与验证
    for epoch in range(start_epoch, args.epochs):
        handler.current_epoch = epoch
        traval.train(epoch)
        loss, top1, top5 = val.validation(epoch)


        saver.save_checkpoint(epoch)
        saver.update_best(top1, top5, epoch)


if __name__ == '__main__':
    main()


