"""
Train a linear probe on top of a frozen ViT-B/16 backbone.

This script trains ONLY the classification head — the backbone is completely
frozen. This establishes the supervised pretrained baseline for comparison
against DINO, MAE, and I-JEPA in later experiments.

Usage (run from repo root):
    python scripts/train_linear_probe.py --dataset pathmnist --seed 0
    python scripts/train_linear_probe.py --dataset pathmnist --seed 1
    python scripts/train_linear_probe.py --dataset pathmnist --seed 2
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import argparse
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.classifier import JEPAClassifier
from src.data.medmnist_loader import DATASET_NUM_CLASSES, DATASET_LOADERS
from src.utils.metrics import compute_accuracy, compute_ece, AverageMeter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    meter = AverageMeter()

    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), n=images.size(0))

    return meter.avg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    model.eval()
    all_outputs = []
    all_labels  = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()
        outputs = model(images)
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)

    acc = compute_accuracy(all_outputs, all_labels)
    ece = compute_ece(all_outputs, all_labels)

    return {"accuracy": acc, "ece": ece}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}  |  Seed: {args.seed}  |  Epochs: {args.epochs}")

    # ── Checkpoint path ────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.dataset}_seed{args.seed}.pth"

    # ── W&B ────────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and args.wandb:
        wandb.init(
            project="JEPA-RobustViT",
            name=f"linear_probe_{args.dataset}_seed{args.seed}",
            config=vars(args),
        )

    # ── Model ──────────────────────────────────────────────────────────────
    num_classes = DATASET_NUM_CLASSES[args.dataset]
    model = JEPAClassifier(num_classes=num_classes).to(device)

    # Verify freezing
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen    = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"Trainable params : {len(trainable)}  |  Frozen params: {len(frozen)}")
    assert len(trainable) == 2, (
        f"Expected exactly 2 trainable params (weight + bias of head), "
        f"got {len(trainable)}: {trainable}"
    )

    # ── Data ───────────────────────────────────────────────────────────────
    loader_fn    = DATASET_LOADERS[args.dataset]
    train_loader = loader_fn(batch_size=args.batch_size, split="train")
    val_loader   = loader_fn(batch_size=args.batch_size, split="val")
    test_loader  = loader_fn(batch_size=args.batch_size, split="test")

    # ── Optimiser + scheduler ──────────────────────────────────────────────
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1:>2}/{args.epochs}"
            f"  loss={train_loss:.4f}"
            f"  val_acc={val_metrics['accuracy']:.2f}%"
            f"  val_ece={val_metrics['ece']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if WANDB_AVAILABLE and args.wandb:
            wandb.log({"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        # Save best checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.head.state_dict(), ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    # ── Final test evaluation ───────────────────────────────────────────────
    print("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.head.load_state_dict(checkpoint)
    test_metrics = evaluate(model, test_loader, device)

    print(f"Test Accuracy : {test_metrics['accuracy']:.2f}%")
    print(f"Test ECE      : {test_metrics['ece']:.4f}")

    if WANDB_AVAILABLE and args.wandb:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    # ── Save results JSON ───────────────────────────────────────────────────
    results = {
        "dataset":      args.dataset,
        "seed":         args.seed,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "best_val_acc": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_ece":      test_metrics["ece"],
        "history":      history,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"linear_probe_{args.dataset}_seed{args.seed}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")
    print(f"Checkpoint saved → {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train linear probe on frozen ViT-B/16 backbone"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=list(DATASET_NUM_CLASSES.keys()),
        help="MedMNIST dataset to train on",
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    args = parser.parse_args()
    main(args)