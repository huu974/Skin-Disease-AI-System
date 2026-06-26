"""Command line and YAML configuration parsing for training."""

import argparse
import sys
from pathlib import Path

import yaml

from model.classification_factory import SUPPORTED_CLASSIFICATION_MODELS
from utils.dataset import TRAIN_AUGMENTATION_PROFILES


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def optimizer_params(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if not text:
        return {}

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f"invalid optimizer params: {value}") from exc

    if isinstance(loaded, dict):
        return loaded
    if loaded is not None and not isinstance(loaded, str):
        raise argparse.ArgumentTypeError(
            "optimizer params must be a YAML/JSON dict or comma-separated key=value pairs"
        )

    result = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "optimizer params must be a YAML/JSON dict or comma-separated key=value pairs"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"invalid optimizer param: {item}")
        result[key] = yaml.safe_load(raw_value.strip())
    return result


def optimizer_param(value):
    text = str(value).strip()
    if "=" not in text:
        raise argparse.ArgumentTypeError("optimizer-param must use key=value format")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("optimizer-param key cannot be empty")
    return key, yaml.safe_load(raw_value.strip())


def _bounded_float(parser, args, name, lower=0.0, upper=1.0):
    value = float(getattr(args, name))
    if not lower <= value <= upper:
        parser.error(f"--{name.replace('_', '-')} must be in [{lower}, {upper}]")
    setattr(args, name, value)


def _positive_float(parser, args, name):
    value = float(getattr(args, name))
    if value <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be > 0")
    setattr(args, name, value)


def _config_arg() -> str | None:
    for index, value in enumerate(sys.argv):
        if value == "--config" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith("--config="):
            return value.split("=", 1)[1]
    return None


def parse():
    parser = argparse.ArgumentParser(description="Skin diseases image classification")

    parser.add_argument("--config", default="config/default.yml", type=str, help="YAML config path")
    parser.add_argument(
        "--model",
        default="efficientnet_b3",
        choices=SUPPORTED_CLASSIFICATION_MODELS,
        type=str,
        help="model name",
    )

    parser.add_argument("--datapath-train", default="./skin diseases/train-new", type=str, help="training dataset path")
    parser.add_argument("--val", default=False, type=str2bool, help="enable validation dataloader")
    parser.add_argument("--datapath-val", default="./skin diseases/val", type=str, help="validation dataset path")
    parser.add_argument("--datapath-test", default="./skin diseases/test", type=str, help="test dataset path")
    parser.add_argument(
        "--train-aug",
        default="strong",
        type=str,
        help=f"training augmentation profile: {', '.join(TRAIN_AUGMENTATION_PROFILES)}",
    )
    parser.add_argument(
        "--mixup-cutmix-prob",
        default=0.5,
        type=float,
        help="probability of applying batch-level mixup/cutmix; set 0 to disable",
    )
    parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup beta distribution alpha")
    parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix beta distribution alpha")
    parser.add_argument(
        "--mixup-prob",
        default=0.5,
        type=float,
        help="when mixup/cutmix is applied, probability of choosing mixup instead of cutmix",
    )
    parser.add_argument("--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("--num-workers", default=8, type=int, help="dataloader worker processes")
    parser.add_argument("--channels-last", default=True, type=str2bool, help="use channels_last memory format")
    parser.add_argument("--save-path", default="./variables", type=str, help="checkpoint output directory")
    parser.add_argument(
        "--experiment-name",
        default="",
        type=str,
        help="experiment directory name for efficientnet_b3; auto uses 01, 02, 03...",
    )
    parser.add_argument("--device", default="auto", type=str, help="training device: auto / cpu / cuda:0 / 0 / mlu:0")
    parser.add_argument("--amp", default=True, type=str2bool, help="use CUDA automatic mixed precision")
    parser.add_argument("--log-interval", default=100, type=int, help="print one training log line every N steps")
    parser.add_argument("--patience", default=15, type=int, help="early stopping patience in validation epochs")
    parser.add_argument("--min-delta", default=0.0, type=float, help="minimum validation Top1 improvement for early stopping")

    parser.add_argument("--weight-decay", "--wd", default=3e-4, type=float, help="weight decay")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="optimizer type")
    parser.add_argument("--momentum", default=None, type=float, help="optimizer momentum, if supported")
    parser.add_argument("--dampening", default=None, type=float, help="SGD dampening, if supported")
    parser.add_argument("--nesterov", default=None, type=str2bool, help="enable Nesterov momentum, if supported")
    parser.add_argument(
        "--optimizer-params",
        default=None,
        type=optimizer_params,
        help="optimizer keyword params as a YAML/JSON dict or key=value pairs",
    )
    parser.add_argument(
        "--optimizer-param",
        dest="optimizer_param_overrides",
        action="append",
        default=None,
        type=optimizer_param,
        help="single optimizer keyword param override, repeatable: --optimizer-param momentum=0.9",
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", default=6e-3, type=float, help="initial learning rate")
    parser.add_argument(
        "--lr-policy",
        default="cosine_lr",
        choices=(
            "cosine",
            "cosine_lr",
            "cosine_annealing",
            "cosine_annealing_lr",
            "multistep",
            "multi_step",
            "multistep_lr",
            "multi_step_lr",
            "constant",
            "constant_lr",
            "fixed",
            "fixed_lr",
            "none",
        ),
        type=str,
        help="learning rate schedule policy",
    )
    parser.add_argument("--warmup-length", default=0, type=int, help="linear warmup epochs")
    parser.add_argument("--lowest-lr", default=1e-5, type=float, help="minimum learning rate")
    parser.add_argument("--lr-steps", default=[], type=int_list, help="comma-separated epochs for multistep lr decay")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="multistep lr decay factor")
    parser.add_argument(
        "--loss",
        default="cross_entropy",
        choices=("cross_entropy", "class_balanced"),
        type=str,
        help="training loss function",
    )
    parser.add_argument(
        "--cb-loss-type",
        default="focal",
        choices=("focal", "sigmoid", "softmax", "cross_entropy"),
        type=str,
        help="base loss used by class-balanced loss",
    )
    parser.add_argument("--cb-beta", default=0.85, type=float, help="class-balanced loss beta")
    parser.add_argument("--cb-gamma", default=2.0, type=float, help="class-balanced focal gamma")
    parser.add_argument("--label-smooth", default=0.0, type=float, help="label smoothing factor")

    parser.add_argument("--logterminal", default=True, type=str2bool, help="mirror logs to terminal")
    parser.add_argument("--resume", default="", type=str, help="checkpoint path for resuming training")

    config_path = Path(_config_arg() or parser.get_default("config"))
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            yaml_dict = yaml.safe_load(file) or {}
        valid_keys = {action.dest for action in parser._actions if action.dest != "help"}
        normalized_config = {}
        for key, value in yaml_dict.items():
            dest = str(key).replace("-", "_")
            if dest in valid_keys:
                normalized_config[dest] = value
        parser.set_defaults(**normalized_config)

    args = parser.parse_args()
    args.optimizer_params = optimizer_params(getattr(args, "optimizer_params", {}))
    for key, value in getattr(args, "optimizer_param_overrides", []) or []:
        args.optimizer_params[key] = value
    delattr(args, "optimizer_param_overrides")
    args.train_aug = str(args.train_aug).strip().lower().replace("-", "_")
    if args.train_aug not in TRAIN_AUGMENTATION_PROFILES:
        parser.error(f"--train-aug must be one of: {', '.join(TRAIN_AUGMENTATION_PROFILES)}")
    _bounded_float(parser, args, "mixup_cutmix_prob")
    _bounded_float(parser, args, "mixup_prob")
    _positive_float(parser, args, "mixup_alpha")
    _positive_float(parser, args, "cutmix_alpha")
    return args
