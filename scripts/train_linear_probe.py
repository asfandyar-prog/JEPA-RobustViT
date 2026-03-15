import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

from src.models.classifier import JEPAClassifier
from src.data.medmnist_loader import get_pathmnist_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

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

model = JEPAClassifier(num_classes=9).to(device)

# --- DEBUG: verify freezing ---
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print("Trainable params:")
for n, s in trainable[:20]:
    print("  ", n, s)
print("Total trainable tensors:", len(trainable))
print("Total frozen tensors:", len([n for n, p in model.named_parameters() if not p.requires_grad]))

optimizer = optim.Adam(model.head.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Data
train_loader = get_pathmnist_loader(batch_size=batch_size, train=True)
test_loader = get_pathmnist_loader(batch_size=batch_size, train=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device).squeeze().long()  # FIX: squeeze and convert to long
        
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
        labels = labels.to(device).squeeze().long()  # FIX: same fix here
        
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Final Test Accuracy: {accuracy:.2f}%")