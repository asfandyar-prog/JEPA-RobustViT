import sys
import os
sys.path.append(os.path.abspath("."))

import torch
from tqdm import tqdm
from src.models.classifier import JEPAClassifier
from src.data.cifar_loader import get_cifar10_loader


device = "cuda" if torch.cuda.is_available() else "cpu"

model = JEPAClassifier(num_classes=10).to(device)
model.eval()

loader = get_cifar10_loader(batch_size=8, train=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")