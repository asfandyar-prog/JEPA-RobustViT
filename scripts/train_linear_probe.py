import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.classifier import JEPAClassifier
from src.data.cifar_loader import get_cifar10_loader


device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
num_epochs = 5
batch_size = 32
learning_rate = 1e-3


# Model
model = JEPAClassifier(num_classes=10).to(device)

# Only train parameters that require gradients (the head)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)

criterion = nn.CrossEntropyLoss()

# Data
train_loader = get_cifar10_loader(batch_size=batch_size, train=True, limit=2000)
test_loader = get_cifar10_loader(batch_size=batch_size, train=False, limit=500)


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