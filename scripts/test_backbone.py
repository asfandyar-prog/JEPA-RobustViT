
import sys
import os
sys.path.append(os.path.abspath("."))

import torch
from src.models.ijepa_backbone import IJEPABackbone

device = "cuda" if torch.cuda.is_available() else "cpu"

model = IJEPABackbone().to(device)
model.eval()

dummy = torch.randn(2, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy)

print("Output shape:", output.shape)
