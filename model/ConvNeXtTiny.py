"""ConvNeXt-Tiny classifier for skin disease classification."""

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = convnext_tiny(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.model.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.model.classifier[-1].bias)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ConvNeXtTinyClassifier(23)
    print(model)