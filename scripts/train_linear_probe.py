# Model - FIX: 9 classes for PathMNIST
model = JEPAClassifier(num_classes=9).to(device)

# ... (rest of your code until the loop)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.squeeze().long()  # FIX: PathMNIST format
        
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
        labels = labels.squeeze().long()  # FIX: PathMNIST format
        
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")