import torch
import torch.nn as nn
from src.models.ijepa_backbone import IJEPABackbone


class JEPAClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.backbone = IJEPABackbone()
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits