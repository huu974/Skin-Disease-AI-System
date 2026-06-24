"""Training entry point for skin disease classification."""

import os
import signal
import sys

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from model.PanDerm import MyModel
from model.ResNet50 import ResNet50Classifier
from model.custom_skin_net import CustomSkinNet
from train_validation import tra_val
from utils.arguments import parse
from utils.config_handler import model_conf
from utils.dataset import get_train_dataloader, get_val_dataloader
from utils.device import device_summary, resolve_device
from utils.optimizer_Adam import CustomAdam
from utils.outputwriter import OutputSave
from utils.writer import init_writer


class InterruptHandler:
    def __init__(self, saver):
        self.saver = saver
        self.current_epoch = 0

    def handler(self, signum, frame):
        print("\nDetected Ctrl+C, saving checkpoint...")
        self.saver.save_checkpoint(self.current_epoch)
        print(f"Checkpoint saved to {self.saver.args.save_path}")
        print("Use --resume next time to continue training.")
        sys.exit(0)


def create_model(model_name: str):
    if model_name == "resnet50":
        return ResNet50Classifier(num_classes=model_conf["num_classes"], pretrained=True)

    if model_name == "efficientnet_b3":
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        return MyModel(model=backbone, num_classes=model_conf["num_classes"]).model_classifier()

    if model_name == "custom_skin_net":
        return CustomSkinNet(
            num_classes=model_conf["num_classes"],
            width_coef=1.5,
            depth_coef=1.4,
            pretrained=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main():
    args = parse()
    device = resolve_device(args.device)
    args.device = str(device)
    if device.type != "cuda":
        args.amp = False

    writer = init_writer(args)
    print(device_summary(device))

    cudnn.benchmark = device.type == "cuda"

    train_dataloader = get_train_dataloader(args)
    val_dataloader = get_val_dataloader(args)

    model_name = getattr(args, "model", "efficientnet_b3")
    os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), model_conf["save_path"])
    args.save_path = os.path.join(args.save_path, model_name)

    model = create_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if device.type == "cuda" else None
    saver = OutputSave(model, args, optimizer)

    start_epoch = 0
    print(f"resume: {args.resume}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resume training from epoch {start_epoch}")

    handler = InterruptHandler(saver)
    signal.signal(signal.SIGINT, handler.handler)

    trainer = tra_val(model, criterion, optimizer, scaler, args, train_dataloader, None, writer, device)
    validator = tra_val(model, criterion, optimizer, scaler, args, None, val_dataloader, writer, device)

    for epoch in range(start_epoch, args.epochs):
        handler.current_epoch = epoch
        trainer.train(epoch)

        top1 = None
        top5 = None
        if val_dataloader is not None:
            _, top1, top5 = validator.validation(epoch)

        saver.save_checkpoint(epoch)
        if top1 is not None and top5 is not None:
            saver.update_best(top1, top5, epoch)


if __name__ == "__main__":
    main()
