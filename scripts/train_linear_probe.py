import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.classifier import JEPAClassifier
from src.data.cifar_loader import get_cifar10_loader


device = "cuda" if torch.cuda.is_available() else "cpu"

import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Hyperparameters
num_epochs = 5
batch_size = 32
learning_rate = 1e-3


seed=2
set_seed(seed)

# Model
model = JEPAClassifier(num_classes=10).to(device)

# --- DEBUG: verify freezing ---
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
frozen = [(n, p.shape) for n, p in model.named_parameters() if not p.requires_grad]

print("Trainable params:")
for n, s in trainable[:20]:
    print("  ", n, s)
print("Total trainable tensors:", len(trainable))
print("Total frozen tensors:", len(frozen))

# Hard assert: ONLY head should be trainable
assert all(n.startswith("head.") for n, _ in trainable), "Backbone is NOT frozen!"
# Only train parameters that require gradients (the head)
optimizer = optim.Adam(model.head.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

# Data
train_loader = get_cifar10_loader(train=True, limit=5000)
test_loader = get_cifar10_loader(train=False, limit=None)


for epoch in range(num_epochs):

    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")


# Evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Final Accuracy: {accuracy:.2f}%")