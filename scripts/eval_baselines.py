"""
Baseline evaluation for JEPA-RobustViT.

Evaluates official pretrained DINO and MAE ViT-B/16 weights using a linear
probe trained on PathMNIST, then runs domain shift evaluation across all
three target domains.

This script uses Option A — official pretrained weights — which is the
standard approach for SSL comparison papers. No retraining of DINO or MAE
is performed.

Pretrained weight sources:
  DINO : https://github.com/facebookresearch/dino
         vit_base_patch16_224 — ViT-B/16 pretrained with DINO on ImageNet
  MAE  : https://github.com/facebookresearch/mae
         vit_base_patch16 — ViT-B/16 pretrained with MAE on ImageNet

Usage (run from repo root):
    python scripts/eval_baselines.py --method dino --seed 0
    python scripts/eval_baselines.py --method dino --seed 1
    python scripts/eval_baselines.py --method dino --seed 2
    python scripts/eval_baselines.py --method mae  --seed 0
    python scripts/eval_baselines.py --method mae  --seed 1
    python scripts/eval_baselines.py --method mae  --seed 2
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.metrics import compute_accuracy, compute_ece, AverageMeter
from src.data.medmnist_loader import (
    DATASET_NUM_CLASSES,
    DATASET_LOADERS,
    get_pathmnist_loader,
    get_dermamnist_loader,
    get_bloodmnist_loader,
    get_retinamnist_loader,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 768  # ViT-B/16 CLS token dimension

# Official DINO ViT-B/16 pretrained on ImageNet
# Loaded via torch.hub from facebookresearch/dino
DINO_REPO = "facebookresearch/dino:main"
DINO_MODEL = "dino_vitb16"

# Official MAE ViT-B/16 pretrained on ImageNet
# Loaded via timm — mae_base_patch16_224 uses official MAE weights
MAE_MODEL = "vit_base_patch16_224.mae"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Backbone loaders
# ---------------------------------------------------------------------------

def load_dino_backbone(device: torch.device) -> nn.Module:
    """
    Load official DINO ViT-B/16 pretrained backbone.
    Returns a module that accepts (B, 3, 224, 224) and outputs (B, 768).

    Uses torch.hub to download official DINO weights from Facebook Research.
    """
    backbone = torch.hub.load(
        DINO_REPO,
        DINO_MODEL,
        pretrained=True,
    )
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device)


def load_mae_backbone(device: torch.device) -> nn.Module:
    """
    Load official MAE ViT-B/16 pretrained backbone via timm.
    Returns a module that accepts (B, 3, 224, 224) and outputs (B, 768).

    timm's mae_base_patch16_224 loads official MAE weights from the
    Facebook Research MAE repository.
    """
    import timm
    backbone = timm.create_model(
        MAE_MODEL,
        pretrained=True,
        num_classes=0,
        global_pool="",
    )
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device)


# ---------------------------------------------------------------------------
# Feature extraction wrappers
# ---------------------------------------------------------------------------

class DINOFeatureExtractor(nn.Module):
    """
    Wraps DINO backbone to return CLS token of shape (B, 768).
    DINO's forward() already returns the CLS token directly.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MAEFeatureExtractor(nn.Module):
    """
    Wraps MAE backbone (timm) to return CLS token of shape (B, 768).
    timm's forward_features returns (B, N+1, 768); index 0 is the CLS token.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return features[:, 0]


# ---------------------------------------------------------------------------
# Linear classifier head
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    """
    Single linear layer classifier trained on top of frozen features.
    Identical architecture to JEPAClassifier.head for fair comparison.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    extractor: nn.Module,
    head: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    """Train the linear head for one epoch. Extractor is always frozen."""
    head.train()
    extractor.eval()
    meter = AverageMeter()

    for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()

        with torch.no_grad():
            features = extractor(images)

        optimizer.zero_grad()
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), n=images.size(0))

    return meter.avg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    extractor: nn.Module,
    head: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    """Evaluate extractor + head on a dataloader. Returns accuracy and ECE."""
    extractor.eval()
    head.eval()

    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()
        features = extractor(images)
        logits = head(features)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "accuracy": compute_accuracy(all_logits, all_labels),
        "ece": compute_ece(all_logits, all_labels),
    }


# ---------------------------------------------------------------------------
# Domain shift evaluation
# ---------------------------------------------------------------------------

def run_domain_shift(
    extractor: nn.Module,
    head: nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict:
    """
    Evaluate model across all four domains.
    Returns per-domain accuracy, ECE, absolute drop, and relative retention.
    """
    domains = [
        ("pathmnist_source",   get_pathmnist_loader(batch_size=batch_size,  split="test")),
        ("dermamnist_shift1",  get_dermamnist_loader(batch_size=batch_size,  split="test")),
        ("bloodmnist_shift2",  get_bloodmnist_loader(batch_size=batch_size,  split="test")),
        ("retinamnist_shift3", get_retinamnist_loader(batch_size=batch_size, split="test")),
    ]

    results = {}
    source_acc = None

    for domain_key, loader in domains:
        print(f"  {domain_key}")
        metrics = evaluate(extractor, head, loader, device)
        results[domain_key] = metrics

        if source_acc is None:
            source_acc = metrics["accuracy"]
        else:
            results[domain_key]["drop"] = source_acc - metrics["accuracy"]
            results[domain_key]["retained_pct"] = (
                100.0 * metrics["accuracy"] / source_acc
            )

    results["pathmnist_source"]["drop"] = 0.0
    results["pathmnist_source"]["retained_pct"] = 100.0

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Method  : {args.method.upper()}")
    print(f"Seed    : {args.seed}")
    print(f"Epochs  : {args.epochs}")

    # ── Load backbone ──────────────────────────────────────────────────────
    print(f"\nLoading {args.method.upper()} pretrained backbone...")
    if args.method == "dino":
        backbone = load_dino_backbone(device)
        extractor = DINOFeatureExtractor(backbone)
    elif args.method == "mae":
        backbone = load_mae_backbone(device)
        extractor = MAEFeatureExtractor(backbone)
    else:
        raise ValueError(f"Unknown method: {args.method}. Choose 'dino' or 'mae'.")

    extractor = extractor.to(device)
    extractor.eval()

    # Verify output dimension
    dummy = torch.zeros(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = extractor(dummy)
    assert out.shape == (2, EMBED_DIM), (
        f"Expected extractor output (2, {EMBED_DIM}), got {out.shape}"
    )
    print(f"Backbone output shape verified: {out.shape}")

    # ── Linear head ────────────────────────────────────────────────────────
    num_classes = DATASET_NUM_CLASSES["pathmnist"]
    head = LinearHead(in_features=EMBED_DIM, num_classes=num_classes).to(device)

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader = get_pathmnist_loader(
        batch_size=args.batch_size, split="train", num_workers=4
    )
    val_loader = get_pathmnist_loader(
        batch_size=args.batch_size, split="val", num_workers=4
    )

    # ── Optimiser + scheduler ──────────────────────────────────────────────
    optimizer = optim.Adam(head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.method}_pathmnist_seed{args.seed}.pth"

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            extractor, head, train_loader,
            optimizer, criterion, device, epoch, args.epochs,
        )
        scheduler.step()
        val_metrics = evaluate(extractor, head, val_loader, device)

        print(
            f"Epoch {epoch + 1:>2}/{args.epochs}"
            f"  loss={train_loss:.4f}"
            f"  val_acc={val_metrics['accuracy']:.2f}%"
            f"  val_ece={val_metrics['ece']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(head.state_dict(), ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    # ── Load best checkpoint for final evaluation ──────────────────────────
    print("\n=== Final Evaluation ===")
    head.load_state_dict(torch.load(ckpt_path, weights_only=True))

    test_loader = get_pathmnist_loader(
        batch_size=args.batch_size, split="test", num_workers=4
    )
    test_metrics = evaluate(extractor, head, test_loader, device)
    print(f"Test Accuracy : {test_metrics['accuracy']:.2f}%")
    print(f"Test ECE      : {test_metrics['ece']:.4f}")

    # ── Domain shift evaluation ────────────────────────────────────────────
    print("\n=== Domain Shift Evaluation ===")
    shift_results = run_domain_shift(extractor, head, device, args.batch_size)

    print(f"\n{'='*60}")
    print(f"{'DOMAIN SHIFT SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Domain':<30} {'Acc':>7} {'Drop':>8} {'Ret%':>6} {'ECE':>7}")
    print(f"{'-'*60}")
    for domain_key, metrics in shift_results.items():
        print(
            f"{domain_key:<30}"
            f" {metrics['accuracy']:>6.2f}%"
            f" {metrics['drop']:>+7.2f}%"
            f" {metrics['retained_pct']:>5.1f}%"
            f" {metrics['ece']:>7.4f}"
        )
    print(f"{'='*60}")

    # ── Save results ───────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output = {
        "method":        args.method,
        "seed":          args.seed,
        "epochs":        args.epochs,
        "lr":            args.lr,
        "batch_size":    args.batch_size,
        "best_val_acc":  best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_ece":      test_metrics["ece"],
        "domain_shift":  shift_results,
    }

    json_path = results_dir / f"baseline_{args.method}_seed{args.seed}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {json_path}")

    txt_path = results_dir / f"baseline_{args.method}_seed{args.seed}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Method : {args.method.upper()}\n")
        f.write(f"Seed   : {args.seed}\n\n")
        f.write(f"{'Domain':<30} {'Acc':>7} {'Drop':>8} {'Ret%':>6} {'ECE':>7}\n")
        f.write(f"{'-'*60}\n")
        for domain_key, metrics in shift_results.items():
            f.write(
                f"{domain_key:<30}"
                f" {metrics['accuracy']:>6.2f}%"
                f" {metrics['drop']:>+7.2f}%"
                f" {metrics['retained_pct']:>5.1f}%"
                f" {metrics['ece']:>7.4f}\n"
            )
    print(f"Results saved → {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DINO or MAE pretrained ViT-B/16 baseline"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["dino", "mae"],
        help="Which pretrained backbone to evaluate",
    )
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save linear head checkpoints",
    )
    args = parser.parse_args()
    main(args)