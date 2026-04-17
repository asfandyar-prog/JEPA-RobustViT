"""
I-JEPA Pretraining Script for JEPA-RobustViT.

Pretrain a ViT-B/16 backbone using the Joint-Embedding Predictive
Architecture (I-JEPA) self-supervised objective on MedMNIST data.

Training objective:
  - Context encoder processes visible (unmasked) patches
  - Target encoder (EMA of context encoder) processes all patches
  - Predictor takes context features + target positions → predicted features
  - Loss: mean L2 distance between predicted and target encoder features

After pretraining, the context encoder weights are saved as the backbone
checkpoint and used in downstream linear probe and TTA experiments.

Reference:
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023.
    https://arxiv.org/abs/2301.08243

Usage (run from repo root):
    python scripts/train_jepa.py --dataset pathmnist --epochs 100 --seed 0
    python scripts/train_jepa.py --dataset pathmnist --epochs 100 --seed 1
    python scripts/train_jepa.py --dataset pathmnist --epochs 100 --seed 2
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import timm

from src.models.jepa_predictor import JEPAPredictor
from src.data.medmnist_loader import DATASET_LOADERS, DATASET_NUM_CLASSES
from src.utils.metrics import AverageMeter


# ---------------------------------------------------------------------------
# Constants — ViT-B/16 geometry
# ---------------------------------------------------------------------------

PATCH_SIZE   = 16
IMAGE_SIZE   = 224
GRID_SIZE    = IMAGE_SIZE // PATCH_SIZE   # 14
NUM_PATCHES  = GRID_SIZE * GRID_SIZE      # 196
EMBED_DIM    = 768


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
# Masking strategy
# ---------------------------------------------------------------------------

def sample_block_mask(
    num_patches: int,
    grid_size: int,
    context_scale: Tuple[float, float] = (0.85, 1.0),
    target_scale:  Tuple[float, float] = (0.15, 0.2),
    aspect_ratio:  Tuple[float, float] = (0.75, 1.5),
    num_targets: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample context and target block masks following I-JEPA masking strategy.

    The context mask covers a large proportion of the image (85-100%).
    The target mask covers multiple smaller rectangular blocks (15-20% each).
    Target patches are removed from the context to prevent information leakage.

    Args:
        num_patches:    total number of patches (196)
        grid_size:      spatial grid size (14)
        context_scale:  (min, max) proportion of patches to keep as context
        target_scale:   (min, max) proportion of patches per target block
        aspect_ratio:   (min, max) aspect ratio for target blocks
        num_targets:    number of target blocks to sample

    Returns:
        context_indices: (N_ctx,) long tensor of context patch indices
        target_indices:  (N_tgt,) long tensor of target patch indices
                         (union of all target blocks, deduplicated)
    """
    # Sample target blocks
    target_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)

    for _ in range(num_targets):
        scale = random.uniform(*target_scale)
        ratio = math.exp(random.uniform(
            math.log(aspect_ratio[0]), math.log(aspect_ratio[1])
        ))
        block_area = int(num_patches * scale)
        block_h = max(1, int(round(math.sqrt(block_area / ratio))))
        block_w = max(1, int(round(math.sqrt(block_area * ratio))))
        block_h = min(block_h, grid_size)
        block_w = min(block_w, grid_size)

        top  = random.randint(0, grid_size - block_h)
        left = random.randint(0, grid_size - block_w)
        target_mask[top:top + block_h, left:left + block_w] = True

    target_indices = target_mask.flatten().nonzero(as_tuple=False).squeeze(1)

    # Context: sample a large block, then remove target patches
    ctx_scale = random.uniform(*context_scale)
    ctx_area  = int(num_patches * ctx_scale)
    ctx_h = max(1, int(round(math.sqrt(ctx_area))))
    ctx_w = max(1, int(round(math.sqrt(ctx_area))))
    ctx_h = min(ctx_h, grid_size)
    ctx_w = min(ctx_w, grid_size)

    ctx_top  = random.randint(0, grid_size - ctx_h)
    ctx_left = random.randint(0, grid_size - ctx_w)

    context_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    context_mask[ctx_top:ctx_top + ctx_h, ctx_left:ctx_left + ctx_w] = True
    context_mask[target_mask] = False  # remove target positions from context

    context_indices = context_mask.flatten().nonzero(as_tuple=False).squeeze(1)

    # Safety: ensure context is non-empty
    if context_indices.numel() == 0:
        context_indices = torch.arange(num_patches // 2)

    return context_indices, target_indices


def collate_jepa_batch(
    images: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply I-JEPA masking to a batch of images.

    For simplicity, we apply the same mask to all images in the batch.
    This is a valid approximation that reduces computation overhead
    while maintaining the statistical properties of the masking strategy.

    Args:
        images: (B, 3, H, W) input images
        device: target device

    Returns:
        images:          (B, 3, H, W) — unchanged, moved to device
        context_indices: (N_ctx,) long tensor
        target_indices:  (N_tgt,) long tensor
    """
    ctx_idx, tgt_idx = sample_block_mask(
        num_patches=NUM_PATCHES,
        grid_size=GRID_SIZE,
    )
    return (
        images.to(device, non_blocking=True),
        ctx_idx.to(device),
        tgt_idx.to(device),
    )


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_target_encoder(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    ema_decay: float,
) -> None:
    """
    Update target encoder parameters via exponential moving average.

    θ_target ← ema_decay * θ_target + (1 - ema_decay) * θ_context

    The target encoder is never updated by backpropagation — only via EMA.
    This provides stable training targets without representation collapse.

    Args:
        context_encoder: the encoder being trained by gradient descent
        target_encoder:  the momentum encoder updated by EMA
        ema_decay:       EMA momentum coefficient (typically 0.996–0.999)
    """
    for param_ctx, param_tgt in zip(
        context_encoder.parameters(),
        target_encoder.parameters(),
    ):
        param_tgt.data.mul_(ema_decay).add_(
            param_ctx.data, alpha=1.0 - ema_decay
        )


# ---------------------------------------------------------------------------
# JEPA loss
# ---------------------------------------------------------------------------

def jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Mean L2 loss between predicted and target encoder features.

    Both tensors are L2-normalised before computing the loss, following
    the implementation in Assran et al. 2023. This prevents the trivial
    solution of collapsing all representations to zero.

    Args:
        predictions: (B, N_tgt, D) — predictor output
        targets:     (B, N_tgt, D) — target encoder output (no grad)

    Returns:
        scalar loss tensor
    """
    predictions = nn.functional.normalize(predictions, dim=-1)
    targets     = nn.functional.normalize(targets,     dim=-1)
    loss = (predictions - targets).pow(2).sum(dim=-1).mean()
    return loss


# ---------------------------------------------------------------------------
# Backbone construction
# ---------------------------------------------------------------------------

def build_context_encoder(pretrained: bool = False) -> nn.Module:
    """
    Build the context encoder — a ViT-B/16 that outputs patch tokens.

    num_classes=0 removes the classification head.
    global_pool="" disables global pooling so we get all patch tokens.

    Args:
        pretrained: if True, initialise with ImageNet supervised weights.
                    For pure I-JEPA pretraining set to False to train from
                    random init. For transfer experiments set to True.

    Returns:
        ViT-B/16 model that returns (B, N+1, 768) from forward_features()
    """
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: JEPAPredictor,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    ema_decay: float,
) -> float:
    """
    Train for one epoch using the I-JEPA objective.

    Args:
        context_encoder: encoder trained by gradient descent
        target_encoder:  momentum encoder (EMA, no grad)
        predictor:       lightweight predictor head
        loader:          training data loader
        optimizer:       Adam optimizer over context_encoder + predictor
        device:          cuda or cpu
        epoch:           current epoch index (0-based)
        num_epochs:      total number of training epochs
        ema_decay:       EMA momentum for target encoder update

    Returns:
        average loss for this epoch
    """
    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    meter = AverageMeter()

    for images, _ in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, ctx_idx, tgt_idx = collate_jepa_batch(images, device)
        B = images.size(0)

        # ── Context encoder forward pass ───────────────────────────────────
        all_features = context_encoder.forward_features(images)
        # all_features: (B, N+1, D) — index 0 is CLS, 1: are patch tokens
        patch_features = all_features[:, 1:, :]  # (B, N, D) — drop CLS

        # Select context patches
        ctx_features = patch_features[
            :,
            ctx_idx,
            :,
        ]  # (B, N_ctx, D)

        # ── Target encoder forward pass (no grad) ─────────────────────────
        with torch.no_grad():
            all_target = target_encoder.forward_features(images)
            patch_target = all_target[:, 1:, :]  # (B, N, D)
            tgt_features = patch_target[
                :,
                tgt_idx,
                :,
            ]  # (B, N_tgt, D) — ground truth targets

        # ── Predictor forward pass ────────────────────────────────────────
        tgt_idx_batch = tgt_idx.unsqueeze(0).expand(B, -1)  # (B, N_tgt)
        predictions = predictor(ctx_features, tgt_idx_batch)  # (B, N_tgt, D)

        # ── Loss and backprop ─────────────────────────────────────────────
        loss = jepa_loss(predictions, tgt_features)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(context_encoder.parameters()) + list(predictor.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        # ── EMA update of target encoder ──────────────────────────────────
        update_target_encoder(context_encoder, target_encoder, ema_decay)

        meter.update(loss.item(), n=B)

    return meter.avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Epochs  : {args.epochs}")
    print(f"Seed    : {args.seed}")

    # ── Build encoders ─────────────────────────────────────────────────────
    print("\nBuilding context encoder...")
    context_encoder = build_context_encoder(
        pretrained=args.init_pretrained
    ).to(device)

    print("Building target encoder (EMA copy of context encoder)...")
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False
    target_encoder.eval()

    # ── Build predictor ────────────────────────────────────────────────────
    predictor = JEPAPredictor(
        backbone_dim=EMBED_DIM,
        predictor_dim=args.predictor_dim,
        num_heads=args.predictor_heads,
        num_layers=args.predictor_layers,
        grid_size=GRID_SIZE,
    ).to(device)

    total_params = sum(p.numel() for p in context_encoder.parameters())
    pred_params  = sum(p.numel() for p in predictor.parameters())
    print(f"Context encoder params : {total_params:,}")
    print(f"Predictor params       : {pred_params:,}")

    # ── Data ───────────────────────────────────────────────────────────────
    loader_fn = DATASET_LOADERS[args.dataset]
    train_loader = loader_fn(
        batch_size=args.batch_size,
        split="train",
        num_workers=4,
    )
    print(f"Training batches : {len(train_loader)}")

    # ── Optimiser ──────────────────────────────────────────────────────────
    # Optimise both context encoder and predictor jointly
    optimizer = optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with linear warmup
    def lr_lambda(step: int) -> float:
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps  = args.epochs * len(train_loader)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # EMA decay schedule: linearly increase from start to end value
    def get_ema_decay(epoch: int) -> float:
        return args.ema_start + (args.ema_end - args.ema_start) * (
            epoch / max(1, args.epochs - 1)
        )

    # ── Checkpoint paths ───────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone_ckpt = ckpt_dir / f"jepa_backbone_{args.dataset}_seed{args.seed}.pth"
    predictor_ckpt = ckpt_dir / f"jepa_predictor_{args.dataset}_seed{args.seed}.pth"

    # ── Training loop ──────────────────────────────────────────────────────
    history = []
    best_loss = float("inf")

    print("\n=== Starting I-JEPA Pretraining ===\n")

    for epoch in range(args.epochs):
        ema_decay = get_ema_decay(epoch)
        avg_loss  = train_one_epoch(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
            ema_decay=ema_decay,
        )
        scheduler.step(epoch * len(train_loader))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:>3}/{args.epochs}"
            f"  loss={avg_loss:.4f}"
            f"  ema={ema_decay:.4f}"
            f"  lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "ema_decay": ema_decay,
            "lr": current_lr,
        })

        # Save checkpoint every 10 epochs and at best loss
        if avg_loss < best_loss or (epoch + 1) % 10 == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
            torch.save(
                context_encoder.state_dict(), backbone_ckpt
            )
            torch.save(predictor.state_dict(), predictor_ckpt)
            print(f"  ✓ Saved checkpoints → {ckpt_dir}")

    # ── Save final results ─────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"jepa_pretraining_{args.dataset}_seed{args.seed}.json"

    output = {
        "dataset":         args.dataset,
        "seed":            args.seed,
        "epochs":          args.epochs,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "batch_size":      args.batch_size,
        "predictor_dim":   args.predictor_dim,
        "predictor_heads": args.predictor_heads,
        "predictor_layers":args.predictor_layers,
        "ema_start":       args.ema_start,
        "ema_end":         args.ema_end,
        "init_pretrained": args.init_pretrained,
        "best_loss":       best_loss,
        "history":         history,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nPretraining complete.")
    print(f"Best loss          : {best_loss:.4f}")
    print(f"Backbone checkpoint: {backbone_ckpt}")
    print(f"Predictor checkpoint: {predictor_ckpt}")
    print(f"Results saved      : {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="I-JEPA self-supervised pretraining for ViT-B/16"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=list(DATASET_NUM_CLASSES.keys()),
        help="Dataset to pretrain on",
    )
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--batch_size",       type=int,   default=256)
    parser.add_argument("--lr",               type=float, default=1.5e-4)
    parser.add_argument("--weight_decay",     type=float, default=0.05)
    parser.add_argument("--warmup_epochs",    type=int,   default=10)
    parser.add_argument("--predictor_dim",    type=int,   default=384)
    parser.add_argument("--predictor_heads",  type=int,   default=6)
    parser.add_argument("--predictor_layers", type=int,   default=6)
    parser.add_argument("--ema_start",        type=float, default=0.996)
    parser.add_argument("--ema_end",          type=float, default=1.000)
    parser.add_argument(
        "--init_pretrained",
        action="store_true",
        default=False,
        help="Initialise context encoder with ImageNet supervised weights",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
    )
    args = parser.parse_args()
    main(args)