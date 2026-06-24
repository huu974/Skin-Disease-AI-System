"""Command line and YAML configuration parsing for training."""

import argparse
import sys
from pathlib import Path

import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


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
    parser.add_argument("--model", default="efficientnet_b3", type=str, help="model name")

    parser.add_argument("--datapath-train", default="./skin diseases/train-new", type=str, help="training dataset path")
    parser.add_argument("--val", default=False, type=str2bool, help="enable validation dataloader")
    parser.add_argument("--datapath-val", default="./skin diseases/val", type=str, help="validation dataset path")
    parser.add_argument("--datapath-test", default="./skin diseases/test", type=str, help="test dataset path")
    parser.add_argument("--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("--channels-last", default=True, type=str2bool, help="use channels_last memory format")
    parser.add_argument("--save-path", default="./variables", type=str, help="checkpoint output directory")
    parser.add_argument("--device", default="auto", type=str, help="training device: auto / cpu / cuda:0 / 0 / mlu:0")
    parser.add_argument("--amp", default=True, type=str2bool, help="use CUDA automatic mixed precision")
    parser.add_argument("--log-interval", default=100, type=int, help="print one training log line every N steps")
    parser.add_argument("--patience", default=15, type=int, help="early stopping patience in validation epochs")
    parser.add_argument("--min-delta", default=0.0, type=float, help="minimum validation Top1 improvement for early stopping")

    parser.add_argument("--weight-decay", "--wd", default=1e-3, type=float, help="weight decay")
    parser.add_argument("--optimizer", default="Adam", type=str, help="optimizer type")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")

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

    return parser.parse_args()
