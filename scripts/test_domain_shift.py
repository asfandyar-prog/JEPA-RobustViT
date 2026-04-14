
import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

from src.models.classifier import JEPAClassifier
from src.data.medmnist_loader import get_pathmnist_loader, get_dermamnist_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return 100 * correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    print(f"Loading model from {args.checkpoint}")
    model = JEPAClassifier(num_classes=9).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.head.load_state_dict(checkpoint)
    else:
        print("Warning: No checkpoint provided, using random weights!")
    
    # Test on PathMNIST (same domain)
    print("\n=== Testing on PathMNIST (source domain) ===")
    path_loader = get_pathmnist_loader(batch_size=args.batch_size, train=False)
    path_acc = evaluate(model, path_loader, device)
    print(f"PathMNIST Accuracy: {path_acc:.2f}%")
    
    # Test on DermaMNIST (target domain - shift)
    print("\n=== Testing on DermaMNIST (target domain) ===")
    derma_loader = get_dermamnist_loader(batch_size=args.batch_size, train=False)
    derma_acc = evaluate(model, derma_loader, device)
    print(f"DermaMNIST Accuracy: {derma_acc:.2f}%")
    
    # Calculate drop
    drop = path_acc - derma_acc
    print(f"\n{'='*40}")
    print(f"DOMAIN SHIFT DROP: {drop:.2f}%")
    print(f"Relative performance: {100 * derma_acc/path_acc:.1f}% of source")
    print(f"{'='*40}")
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            f.write(f"Source domain (PathMNIST): {path_acc:.2f}%\n")
            f.write(f"Target domain (DermaMNIST): {derma_acc:.2f}%\n")
            f.write(f"Absolute drop: {drop:.2f}%\n")
            f.write(f"Relative performance: {100 * derma_acc/path_acc:.1f}%\n")
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_pathmnist.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_results", type=str, default="domain_shift_results.txt",
                       help="File to save results")
    args = parser.parse_args()
    
    main(args)
