#!/bin/bash
#SBATCH --job-name=jepa_pretrain
#SBATCH --output=logs/jepa_pretrain_%j.out
#SBATCH --error=logs/jepa_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# I-JEPA Pretraining — SLURM Job Script
# University of Debrecen HPC
#
# Usage:
#   sbatch hpc/train_jepa.sh                    # seed 0 (default)
#   sbatch hpc/train_jepa.sh --export=SEED=1    # seed 1
#   sbatch hpc/train_jepa.sh --export=SEED=2    # seed 2
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Environment setup ──────────────────────────────────────────────────────
echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "GPUs         : $SLURM_GRES"
echo "Start time   : $(date)"
echo "========================================"

# Load modules (adjust module names to match Unideb HPC environment)
module load python/3.11 2>/dev/null || true
module load cuda/12.1    2>/dev/null || true

# Activate virtual environment
VENV_PATH="${HOME}/venvs/jepa"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_PATH"
fi
source "${VENV_PATH}/bin/activate"

# Install dependencies if needed
pip install -q -r requirements.txt

# ── Repo setup ─────────────────────────────────────────────────────────────
REPO_DIR="${HOME}/JEPA-RobustViT"
cd "$REPO_DIR"
git pull origin main

# Create output directories
mkdir -p logs checkpoints results

# ── Parameters ────────────────────────────────────────────────────────────
SEED=${SEED:-0}
DATASET=${DATASET:-pathmnist}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1.5e-4}

echo "Dataset    : $DATASET"
echo "Seed       : $SEED"
echo "Epochs     : $EPOCHS"
echo "Batch size : $BATCH_SIZE"
echo "LR         : $LR"
echo ""

# ── Run pretraining ────────────────────────────────────────────────────────
python scripts/train_jepa.py \
    --dataset       "$DATASET"    \
    --seed          "$SEED"       \
    --epochs        "$EPOCHS"     \
    --batch_size    "$BATCH_SIZE" \
    --lr            "$LR"         \
    --weight_decay  0.05          \
    --warmup_epochs 10            \
    --predictor_dim    384        \
    --predictor_heads  6          \
    --predictor_layers 6          \
    --ema_start 0.996             \
    --ema_end   1.000             \
    --checkpoint_dir checkpoints

echo ""
echo "Pretraining complete: $(date)"