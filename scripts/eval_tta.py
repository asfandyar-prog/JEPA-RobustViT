"""
Test-Time Adaptation Evaluation for JEPA-RobustViT.

Evaluates the effect of entropy-based TTA on all domain shift pairs.
Compares four conditions for a given backbone checkpoint:
  1. No TTA  — standard inference
  2. TTA     — entropy minimisation over LayerNorm parameters

Reports accuracy and ECE for each condition across all domain shift pairs.

This script can evaluate any checkpoint saved by:
  - scripts/train_linear_probe.py  (supervised ViT baseline)
  - scripts/eval_baselines.py      (DINO / MAE baselines)
  - A JEPA backbone + linear head  (after train_jepa.py + train_linear_probe.py)

Usage (run from repo root):
    python scripts/eval_tta.py \\
        --checkpoint checkpoints/pathmnist_seed0.pth \\
        --method supervised \\
        --seed 0

    python scripts/eval_tta.py \\
        --checkpoint checkpoints/dino_pathmnist_seed0.pth \\
        --method dino \\
        --seed 0

    python scripts/eval_tta.py \\
        --checkpoint checkpoints/jepa_linear_pathmnist_seed0.pth \\
        --method jepa \\
        --seed 0
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.classifier import JEPAClassifier
from src.models.tta import TTAWrapper
from src.data.medmnist_loader import (
    DATASET_NUM_CLASSES,
    get_pathmnist_loader,
    get_dermamnist_loader,
    get_bloodmnist_loader,
    get_retinamnist_loader,
)
from src.utils.metrics import compute_accuracy, compute_ece


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
# Standard evaluation (no TTA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_standard(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Standard inference without TTA.

    Args:
        model:  trained JEPAClassifier
        loader: test dataloader
        device: cuda or cpu

    Returns:
        dict with 'accuracy' and 'ece'
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  No TTA", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()
        logits = model(images)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "accuracy": compute_accuracy(all_logits, all_labels),
        "ece":      compute_ece(all_logits, all_labels),
    }


# ---------------------------------------------------------------------------
# TTA evaluation
# ---------------------------------------------------------------------------

def evaluate_tta(
    tta_model: TTAWrapper,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Inference with test-time adaptation.

    Each batch is processed through the TTAWrapper which adapts LayerNorm
    parameters before returning predictions.

    Args:
        tta_model: TTAWrapper wrapping a trained JEPAClassifier
        loader:    test dataloader
        device:    cuda or cpu

    Returns:
        dict with 'accuracy' and 'ece'
    """
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  TTA", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze().long()

        logits = tta_model(images)
        all_logits.append(logits.detach())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "accuracy": compute_accuracy(all_logits, all_labels),
        "ece":      compute_ece(all_logits, all_labels),
    }


# ---------------------------------------------------------------------------
# Domain evaluation helper
# ---------------------------------------------------------------------------

def evaluate_all_domains(
    model: nn.Module,
    tta_model: TTAWrapper,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict]:
    """
    Evaluate both standard and TTA inference across all four domains.

    Args:
        model:      trained JEPAClassifier (for standard inference)
        tta_model:  TTAWrapper (for TTA inference)
        device:     cuda or cpu
        batch_size: batch size for all loaders

    Returns:
        nested dict: domain_key → {"no_tta": {...}, "tta": {...}}
    """
    domains = [
        ("pathmnist_source",   get_pathmnist_loader(batch_size=batch_size,  split="test")),
        ("dermamnist_shift1",  get_dermamnist_loader(batch_size=batch_size,  split="test")),
        ("bloodmnist_shift2",  get_bloodmnist_loader(batch_size=batch_size,  split="test")),
        ("retinamnist_shift3", get_retinamnist_loader(batch_size=batch_size, split="test")),
    ]

    results = {}
    source_acc_no_tta = None
    source_acc_tta    = None

    for domain_key, loader in domains:
        print(f"\n  Domain: {domain_key}")

        no_tta_metrics = evaluate_standard(model,     loader, device)
        tta_metrics    = evaluate_tta(tta_model, loader, device)

        results[domain_key] = {
            "no_tta": no_tta_metrics,
            "tta":    tta_metrics,
        }

        if source_acc_no_tta is None:
            source_acc_no_tta = no_tta_metrics["accuracy"]
            source_acc_tta    = tta_metrics["accuracy"]
            results[domain_key]["no_tta"]["drop"]         = 0.0
            results[domain_key]["no_tta"]["retained_pct"] = 100.0
            results[domain_key]["tta"]["drop"]            = 0.0
            results[domain_key]["tta"]["retained_pct"]    = 100.0
        else:
            results[domain_key]["no_tta"]["drop"] = (
                source_acc_no_tta - no_tta_metrics["accuracy"]
            )
            results[domain_key]["no_tta"]["retained_pct"] = (
                100.0 * no_tta_metrics["accuracy"] / source_acc_no_tta
            )
            results[domain_key]["tta"]["drop"] = (
                source_acc_tta - tta_metrics["accuracy"]
            )
            results[domain_key]["tta"]["retained_pct"] = (
                100.0 * tta_metrics["accuracy"] / source_acc_tta
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Method     : {args.method}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"TTA steps  : {args.tta_steps}")
    print(f"TTA lr     : {args.tta_lr}")
    print(f"Episodic   : {args.episodic}")

    # ── Load model ─────────────────────────────────────────────────────────
    num_classes = DATASET_NUM_CLASSES["pathmnist"]
    model = JEPAClassifier(num_classes=num_classes).to(device)

    ckpt = torch.load(args.checkpoint, weights_only=True)
    model.head.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Create TTA wrapper ─────────────────────────────────────────────────
    tta_model = TTAWrapper(
        model=model,
        lr=args.tta_lr,
        steps=args.tta_steps,
        episodic=args.episodic,
        entropy_threshold=args.entropy_threshold if args.entropy_threshold > 0 else None,
    ).to(device)

    print(f"\nTTA model: {tta_model}")
    print(f"Adaptable params: {tta_model.num_adaptable_params:,}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n=== Evaluating across all domains ===")
    results = evaluate_all_domains(model, tta_model, device, args.batch_size)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'TTA EVALUATION SUMMARY':^72}")
    print(f"{'='*72}")
    print(
        f"{'Domain':<28}"
        f" {'NoTTA Acc':>10} {'TTA Acc':>8}"
        f" {'Recovery':>9}"
        f" {'NoTTA ECE':>10} {'TTA ECE':>8}"
    )
    print(f"{'-'*72}")

    for domain_key, metrics in results.items():
        no_tta_acc = metrics["no_tta"]["accuracy"]
        tta_acc    = metrics["tta"]["accuracy"]
        recovery   = tta_acc - no_tta_acc
        no_tta_ece = metrics["no_tta"]["ece"]
        tta_ece    = metrics["tta"]["ece"]

        print(
            f"{domain_key:<28}"
            f" {no_tta_acc:>9.2f}%"
            f" {tta_acc:>7.2f}%"
            f" {recovery:>+8.2f}%"
            f" {no_tta_ece:>10.4f}"
            f" {tta_ece:>8.4f}"
        )

    print(f"{'='*72}")

    # ── Save results ───────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    ckpt_stem = Path(args.checkpoint).stem
    json_path = results_dir / f"tta_{args.method}_{ckpt_stem}_steps{args.tta_steps}.json"
    txt_path  = results_dir / f"tta_{args.method}_{ckpt_stem}_steps{args.tta_steps}.txt"

    output = {
        "method":            args.method,
        "checkpoint":        args.checkpoint,
        "seed":              args.seed,
        "tta_lr":            args.tta_lr,
        "tta_steps":         args.tta_steps,
        "episodic":          args.episodic,
        "entropy_threshold": args.entropy_threshold,
        "domains":           results,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    with open(txt_path, "w") as f:
        f.write(f"Method     : {args.method}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"TTA steps  : {args.tta_steps}\n")
        f.write(f"TTA lr     : {args.tta_lr}\n\n")
        f.write(
            f"{'Domain':<28}"
            f" {'NoTTA Acc':>10} {'TTA Acc':>8}"
            f" {'Recovery':>9}"
            f" {'NoTTA ECE':>10} {'TTA ECE':>8}\n"
        )
        f.write(f"{'-'*72}\n")
        for domain_key, metrics in results.items():
            no_tta_acc = metrics["no_tta"]["accuracy"]
            tta_acc    = metrics["tta"]["accuracy"]
            recovery   = tta_acc - no_tta_acc
            no_tta_ece = metrics["no_tta"]["ece"]
            tta_ece    = metrics["tta"]["ece"]
            f.write(
                f"{domain_key:<28}"
                f" {no_tta_acc:>9.2f}%"
                f" {tta_acc:>7.2f}%"
                f" {recovery:>+8.2f}%"
                f" {no_tta_ece:>10.4f}"
                f" {tta_ece:>8.4f}\n"
            )

    print(f"\nResults saved → {json_path}")
    print(f"Results saved → {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TTA under medical domain shift"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained linear head checkpoint (.pth)",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["supervised", "dino", "mae", "jepa"],
        help="Which backbone was used to train this checkpoint",
    )
    parser.add_argument("--seed",              type=int,   default=0)
    parser.add_argument("--batch_size",        type=int,   default=64)
    parser.add_argument("--tta_lr",            type=float, default=1e-4)
    parser.add_argument("--tta_steps",         type=int,   default=1)
    parser.add_argument(
        "--episodic",
        action="store_true",
        default=True,
        help="Reset model weights after each batch (episodic TTA)",
    )
    parser.add_argument(
        "--no_episodic",
        dest="episodic",
        action="store_false",
        help="Accumulate adaptations across batches (continual TTA)",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.0,
        help="Only adapt samples with entropy above this value. 0 = adapt all.",
    )
    args = parser.parse_args()
    main(args)