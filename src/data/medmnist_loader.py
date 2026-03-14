import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.jiepa_backbone import JIEPABackbone
from src.models.classifier import LinearClassifier
from src.data.pathmnist_loader import get_pathmnist_loader  # Update import


def train_epoch(model, classifier, loader, criterion, optimizer, device):
    model.eval()  # Backbone frozen
    classifier.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training"):
        # PathMNIST returns (images, labels) where labels are [batch_size, 1]
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze()  # Remove extra dimension: [batch_size, 1] -> [batch_size]
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = model(images)
        
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return total_loss / len(loader), all_preds, all_labels


def evaluate(model, classifier, loader, criterion, device):
    model.eval()
    classifier.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze()
            
            features = model(images)
            outputs = classifier(features)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return total_loss / len(loader), all_preds, all_labels


def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load backbone (frozen)
    backbone = JIEPABackbone(pretrained=True).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Classifier: PathMNIST has 9 classes
    classifier = LinearClassifier(
        input_dim=768,
        num_classes=9  # PathMNIST classes: 9 tissue types
    ).to(device)
    
    print(f"Trainable params: {sum(p.numel() for p in classifier.parameters())}")
    
    # Data loaders
    train_loader = get_pathmnist_loader(
        batch_size=args.batch_size,
        train=True
    )
    
    val_loader = get_pathmnist_loader(
        batch_size=args.batch_size,
        train=False  # PathMNIST uses 'test' as validation split
    )
    
    # Loss: CrossEntropy for multi-class
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_preds, train_labels = train_epoch(
            backbone, classifier, train_loader, criterion, optimizer, device
        )
        train_metrics = compute_metrics(train_preds, train_labels)
        
        val_loss, val_preds, val_labels = evaluate(
            backbone, classifier, val_loader, criterion, device
        )
        val_metrics = compute_metrics(val_preds, val_labels)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_metrics': val_metrics
            }, args.save_path)
            print(f"✓ Saved best model with accuracy: {best_acc:.4f}")
    
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    
    # Load best model and run final evaluation
    checkpoint = torch.load(args.save_path)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Optional: Run test evaluation
    if args.test:
        test_loader = get_pathmnist_loader(
            batch_size=args.batch_size,
            train=False  # Using same as validation for now
        )
        test_loss, test_preds, test_labels = evaluate(
            backbone, classifier, test_loader, criterion, device
        )
        test_metrics = compute_metrics(test_preds, test_labels)
        print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="best_pathmnist.pth")
    parser.add_argument("--test", action="store_true", help="Run test evaluation")
    args = parser.parse_args()
    
    main(args)