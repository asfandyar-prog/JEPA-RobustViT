import torch
import torch.nn as nn
from src.models.ijepa_backbone import IJEPABackbone


class JEPAClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.backbone = IJEPABackbone()

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        logits = self.head(features)
        return logits