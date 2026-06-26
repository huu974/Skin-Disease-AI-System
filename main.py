"""Training entry point for skin disease classification."""

import os

import torch
import torch.backends.cudnn as cudnn
import yaml

from model.classification_factory import create_classification_model
from train_validation import tra_val
from utils.arguments import parse
from utils.config_handler import model_conf
from utils.dataset import get_train_dataloader, get_val_dataloader
from utils.device import device_summary, resolve_device
from utils.losses import build_loss
from utils.metric_plotter import MetricPlotter
from utils.optimizer_factory import build_optimizer
from utils.outputwriter import OutputSave
from utils.writer import init_writer


def create_model(model_name: str):
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', model_conf["save_path"])
    return create_classification_model(model_name, num_classes=model_conf["num_classes"], pretrained=True)


def loss_output_name(args):
    loss_name = getattr(args, "loss", "cross_entropy")
    if loss_name == "class_balanced":
        return f"class_balanced_{getattr(args, 'cb_loss_type', 'focal')}"
    return loss_name


def current_loss_name(args):
    loss_name = getattr(args, "loss", "cross_entropy")
    if loss_name == "class_balanced":
        return (
            f"class_balanced:"
            f"{getattr(args, 'cb_loss_type', 'focal')},"
            f"beta={getattr(args, 'cb_beta', 0.9999)},"
            f"gamma={getattr(args, 'cb_gamma', 2.0)},"
            f"label_smooth={getattr(args, 'label_smooth', 0.0)}"
        )
    return f"{loss_name}:label_smooth={getattr(args, 'label_smooth', 0.0)}"


def _validate_experiment_name(experiment_name: str) -> str:
    name = str(experiment_name or "").strip()
    if not name:
        return ""
    invalid_name = (
        name in {".", ".."}
        or os.path.isabs(name)
        or "/" in name
        or "\\" in name
        or os.path.basename(name) != name
    )
    if invalid_name:
        raise ValueError("--experiment-name must be a directory name, not a path")
    return name


def _reserve_numbered_experiment_dir(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    index = 1
    while True:
        path = os.path.join(root, f"{index:02d}")
        try:
            os.makedirs(path, exist_ok=False)
            return path
        except FileExistsError:
            index += 1


def resolve_save_path(args, model_name: str) -> str:
    if model_name == "efficientnet_b3":
        experiment_root = os.path.join(args.save_path, loss_output_name(args))
        experiment_name = _validate_experiment_name(getattr(args, "experiment_name", ""))

        if experiment_name:
            save_path = os.path.join(experiment_root, experiment_name)
            if os.path.exists(save_path) and not getattr(args, "resume", ""):
                raise FileExistsError(
                    f"Experiment directory already exists: {save_path}. "
                    "Use another --experiment-name or resume from this experiment."
                )
            os.makedirs(save_path, exist_ok=True)
            return save_path

        return _reserve_numbered_experiment_dir(experiment_root)

    output_parts = [args.save_path, model_name]
    if getattr(args, "loss", "cross_entropy") != "cross_entropy":
        output_parts.append(loss_output_name(args))
    save_path = os.path.join(*output_parts)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def _yaml_key(name: str) -> str:
    return name.replace("_", "-")


def save_resolved_config(args, base_save_path: str) -> str:
    config = {}
    for key, value in vars(args).items():
        config[_yaml_key(key)] = base_save_path if key == "save_path" else value
    config["resolved-save-path"] = args.save_path

    config_path = os.path.join(args.save_path, "resolved_config.yaml")
    tmp_path = f"{config_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    os.replace(tmp_path, config_path)
    return config_path


def main():
    args = parse()
    device = resolve_device(args.device)
    args.device = str(device)
    if device.type != "cuda":
        args.amp = False

    model_name = getattr(args, "model", "efficientnet_b3")
    base_save_path = args.save_path
    args.save_path = resolve_save_path(args, model_name)
    resolved_config_path = save_resolved_config(args, base_save_path)

    writer = init_writer(args)
    print(f"resolved_config: {resolved_config_path}")
    print(device_summary(device))

    cudnn.benchmark = device.type == "cuda"

    train_dataloader = get_train_dataloader(args)
    val_dataloader = get_val_dataloader(args)

    os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), model_conf["save_path"])
    print(f"save_path: {args.save_path}")

    model = create_model(model_name).to(device)
    criterion = build_loss(args, train_dataloader.dataset, model_conf["num_classes"]).to(device)
    optimizer = build_optimizer(model.parameters(), args)
    print(f"optimizer: {args.optimizer}, params: {args.optimizer_params}")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if device.type == "cuda" else None
    saver = OutputSave(model, args, optimizer)
    metric_plotter = MetricPlotter(args.save_path, resume=bool(args.resume))

    start_epoch = 0
    best_early_stop_top1 = 0.0
    best_epoch = None
    no_improve_epochs = 0
    print(f"resume: {args.resume}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        saver.best_top1 = checkpoint.get("best_top1", 0.0)
        saver.best_top5 = checkpoint.get("best_top5", 0.0)
        saver.best_auc = checkpoint.get("best_auc")
        saver.best_epoch = checkpoint.get("best_epoch")
        saver.no_improve_epochs = checkpoint.get("no_improve_epochs", 0)
        best_early_stop_top1 = saver.best_top1
        best_epoch = saver.best_epoch
        no_improve_epochs = saver.no_improve_epochs
        print(f"Resume training from epoch {start_epoch}")

    trainer = tra_val(model, criterion, optimizer, scaler, args, train_dataloader, None, writer, device)
    validator = tra_val(model, criterion, optimizer, scaler, args, None, val_dataloader, writer, device)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = trainer.train(epoch)

        top1 = None
        top5 = None
        improved = False
        val_metrics = None
        if val_dataloader is not None:
            val_metrics = validator.validation(epoch)
            top1 = val_metrics["top1"]
            top5 = val_metrics["top5"]
            improved = top1 > best_early_stop_top1 + args.min_delta
            if improved:
                best_early_stop_top1 = top1
                best_epoch = epoch + 1
                no_improve_epochs = 0
                saver.no_improve_epochs = no_improve_epochs
                saver.update_best(top1, top5, epoch, val_metrics.get("auc"))
            else:
                no_improve_epochs += 1
                print(
                    f"Early stopping patience: {no_improve_epochs}/{args.patience} "
                    f"(current Top1={top1:.4f}, best Top1={best_early_stop_top1:.4f}, "
                    f"min_delta={args.min_delta})"
                )

        saver.no_improve_epochs = no_improve_epochs
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"] if val_metrics else None,
            "train_top1": train_metrics["top1"],
            "val_top1": val_metrics["top1"] if val_metrics else None,
            "train_top5": train_metrics["top5"],
            "val_top5": val_metrics["top5"] if val_metrics else None,
            "train_auc": train_metrics["auc"],
            "val_auc": val_metrics["auc"] if val_metrics else None,
            "lr": train_metrics["lr"],
            "train_fps": train_metrics["fps"],
            "val_fps": val_metrics["fps"] if val_metrics else None,
        }
        metric_plotter.update(
            epoch_metrics,
            {
                "best_epoch": best_epoch,
                "best_top1": saver.best_top1,
                "best_top5": saver.best_top5,
                "best_auc": saver.best_auc,
                "model": model_name,
                "loss": current_loss_name(args),
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "train_aug": getattr(args, "train_aug", "strong"),
                "mixup_cutmix_prob": getattr(args, "mixup_cutmix_prob", 0.5),
                "mixup_alpha": getattr(args, "mixup_alpha", 0.2),
                "cutmix_alpha": getattr(args, "cutmix_alpha", 1.0),
                "mixup_prob": getattr(args, "mixup_prob", 0.5),
                "lr": args.lr,
                "lr_policy": args.lr_policy,
                "warmup_length": args.warmup_length,
                "lowest_lr": args.lowest_lr,
                "lr_steps": args.lr_steps,
                "lr_gamma": args.lr_gamma,
                "save_path": args.save_path,
                "best_model_path": os.path.join(args.save_path, "best_model.pth.tar"),
            },
        )
        saver.save_checkpoint(epoch)
        if top1 is not None and top5 is not None:
            if no_improve_epochs >= args.patience:
                print(
                    f"Early stopping triggered after epoch {epoch + 1} "
                    f"(no Top1 improvement for {no_improve_epochs} epochs, patience={args.patience})"
                )
                break


if __name__ == "__main__":
    main()
