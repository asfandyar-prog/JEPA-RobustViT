"""
Domain shift evaluation for JEPA-RobustViT.

Evaluates a model trained on PathMNIST (source domain) against
three target domains of increasing shift severity:

  Shift 1: PathMNIST  → DermaMNIST   (tissue → skin lesion)
  Shift 2: PathMNIST  → BloodMNIST   (tissue → blood cell)
  Shift 3: PathMNIST  → RetinaMNIST  (tissue → retinal fundus)

Metrics reported per domain:
  - Top-1 Accuracy
  - Expected Calibration Error (ECE)
  - Absolute accuracy drop vs source
  - Relative performance vs source (%)

Usage (run from repo root):
    python scripts/eval_domain_shift.py --checkpoint checkpoints/pathmnist_seed0.pth
    python scripts/eval_domain_shift.py --checkpoint checkpoints/pathmnist_seed1.pth
    python scripts/eval_domain_shift.py --checkpoint checkpoints/pathmnist_seed2.pth
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.classifier import JEPAClassifier
from src.data.medmnist_loader import (
    get_pathmnist_loader,
    get_dermamnist_loader,
    get_bloodmnist_loader,
    get_retinamnist_loader,
    DATASET_NUM_CLASSES,
)
from src.utils.metrics import compute_accuracy, compute_ece


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    domain_name: str,
) -> dict:
    model.eval()
    all_outputs = []
    all_labels  = []

    for images, labels in tqdm(loader, desc=f"  Evaluating {domain_name}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()
        outputs = model(images)
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)

    return {
        "accuracy": compute_accuracy(all_outputs, all_labels),
        "ece":      compute_ece(all_outputs, all_labels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # ── Load model trained on PathMNIST (9 classes) ────────────────────────
    model = JEPAClassifier(num_classes=DATASET_NUM_CLASSES["pathmnist"]).to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True)
    model.head.load_state_dict(ckpt)
    model.eval()
    print("Model loaded successfully.\n")

    # ── Define evaluation domains ──────────────────────────────────────────
    # Source domain is always PathMNIST test split
    # Target domains are test splits of shift datasets
    domains = [
        ("PathMNIST (source)",          get_pathmnist_loader(split="test",  batch_size=args.batch_size)),
        ("DermaMNIST  (shift target 1)", get_dermamnist_loader(split="test", batch_size=args.batch_size)),
        ("BloodMNIST  (shift target 2)", get_bloodmnist_loader(split="test", batch_size=args.batch_size)),
        ("RetinaMNIST (shift target 3)", get_retinamnist_loader(split="test",batch_size=args.batch_size)),
    ]

    # ── Evaluate all domains ───────────────────────────────────────────────
    domain_results = {}
    for domain_name, loader in domains:
        print(f"── {domain_name}")
        metrics = evaluate_loader(model, loader, device, domain_name)
        domain_results[domain_name] = metrics
        print(f"   Accuracy: {metrics['accuracy']:.2f}%  |  ECE: {metrics['ece']:.4f}")

    # ── Compute shift statistics ───────────────────────────────────────────
    source_acc = domain_results["PathMNIST (source)"]["accuracy"]

    print(f"\n{'='*60}")
    print(f"{'DOMAIN SHIFT SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Domain':<35} {'Acc':>7} {'Drop':>7} {'Rel%':>7} {'ECE':>7}")
    print(f"{'-'*60}")

    summary = {}
    for domain_name, metrics in domain_results.items():
        acc  = metrics["accuracy"]
        drop = source_acc - acc
        rel  = 100.0 * acc / source_acc
        ece  = metrics["ece"]

        summary[domain_name] = {
            "accuracy": acc,
            "drop":     drop,
            "relative": rel,
            "ece":      ece,
        }
        print(f"{domain_name:<35} {acc:>6.2f}% {drop:>+6.2f}% {rel:>6.1f}% {ece:>7.4f}")

    print(f"{'='*60}\n")

    # ── Save results ───────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Extract seed from checkpoint filename e.g. pathmnist_seed0.pth → seed0
    ckpt_stem = Path(args.checkpoint).stem          # e.g. pathmnist_seed0
    results_path = results_dir / f"domain_shift_{ckpt_stem}.json"

    output = {
        "checkpoint": args.checkpoint,
        "source_accuracy": source_acc,
        "domains": summary,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also save human-readable txt
    txt_path = results_dir / f"domain_shift_{ckpt_stem}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write(f"{'Domain':<35} {'Acc':>7} {'Drop':>7} {'Rel%':>7} {'ECE':>7}\n")
        f.write(f"{'-'*60}\n")
        for domain_name, s in summary.items():
            f.write(
                f"{domain_name:<35} {s['accuracy']:>6.2f}%"
                f" {s['drop']:>+6.2f}%"
                f" {s['relative']:>6.1f}%"
                f" {s['ece']:>7.4f}\n"
            )

    print(f"Results saved → {results_path}")
    print(f"Results saved → {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate domain shift: PathMNIST → DermaMNIST / BloodMNIST / RetinaMNIST"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PathMNIST head checkpoint (.pth)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)