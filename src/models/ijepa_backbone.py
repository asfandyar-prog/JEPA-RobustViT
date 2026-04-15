import torch
import torch.nn as nn
import timm


class IJEPABackbone(nn.Module):
    """
    Wrapper for a pretrained Vision Transformer backbone.
    Returns CLS token representation of shape (batch_size, 768).
    """

    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

    def forward(self, x):
        features = self.model.forward_features(x)
        cls_token = features[:, 0]
        return cls_token