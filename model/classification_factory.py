"""Factory helpers for image classification models and checkpoints."""

import os
from pathlib import Path

from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from model.ConvNeXtTiny import ConvNeXtTinyClassifier
from model.PanDerm import MyModel
from model.ResNet50 import ResNet50Classifier
from model.custom_skin_net import CustomSkinNet
from utils.config_handler import model_conf, test_evaluate_conf
from utils.path_tool import get_abs_path


SUPPORTED_CLASSIFICATION_MODELS = (
    "efficientnet_b3",
    "resnet50",
    "custom_skin_net",
    "convnext_tiny",
)


MODEL_DISPLAY_NAMES = {
    "efficientnet_b3": "EfficientNet-B3",
    "resnet50": "ResNet50",
    "custom_skin_net": "Custom Skin Net",
    "convnext_tiny": "ConvNeXt-Tiny",
}


def create_classification_model(model_name: str, num_classes: int | None = None, pretrained: bool = True):
    """Create a supported skin disease classification model."""
    num_classes = num_classes or model_conf["num_classes"]
    os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_conf["save_path"])

    if model_name == "efficientnet_b3":
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        return MyModel(model=backbone, num_classes=num_classes).model_classifier()

    if model_name == "resnet50":
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)

    if model_name == "custom_skin_net":
        return CustomSkinNet(
            num_classes=num_classes,
            width_coef=1.5,
            pretrained=False,
        )

    if model_name == "convnext_tiny":
        return ConvNeXtTinyClassifier(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unsupported model: {model_name}")


def _configured_checkpoint_path(model_name: str) -> str | None:
    config_key = f"classification_model_{model_name}"
    if config_key in test_evaluate_conf:
        return test_evaluate_conf[config_key]

    model_paths = test_evaluate_conf.get("classification_models", {})
    if isinstance(model_paths, dict) and model_name in model_paths:
        return model_paths[model_name]

    if model_name == "efficientnet_b3":
        return test_evaluate_conf.get("classification_model")

    if model_name == "custom_skin_net":
        return test_evaluate_conf.get("classification_model_custom")

    return None


def get_checkpoint_path(model_name: str) -> str:
    """Return the preferred checkpoint path for a model."""
    configured_path = _configured_checkpoint_path(model_name)
    if configured_path:
        return get_abs_path(configured_path)

    return get_abs_path(str(Path("variables") / model_name / "best_model.pth.tar"))


def find_checkpoint_path(model_name: str) -> str:
    """Return an existing checkpoint path when available, otherwise the preferred path."""
    candidates = [
        get_checkpoint_path(model_name),
        get_abs_path(str(Path("variables") / model_name / "best_model.pth.tar")),
        get_abs_path(str(Path("variables") / "best_model.pth.tar")),
    ]

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path

    return candidates[0]
