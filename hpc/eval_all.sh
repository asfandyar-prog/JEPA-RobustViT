#!/bin/bash
#SBATCH --job-name=jepa_eval
#SBATCH --output=logs/jepa_eval_%j.out
#SBATCH --error=logs/jepa_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Full Evaluation Pipeline — SLURM Job Script
# Runs linear probe + domain shift + TTA for a given method and seed.
#
# Usage:
#   sbatch hpc/eval_all.sh
#   sbatch hpc/eval_all.sh --export=METHOD=dino,SEED=0
#   sbatch hpc/eval_all.sh --export=METHOD=jepa,SEED=1
# ---------------------------------------------------------------------------

set -euo pipefail

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Start time : $(date)"
echo "========================================"

module load python/3.11 2>/dev/null || true
module load cuda/12.1    2>/dev/null || true

VENV_PATH="${HOME}/venvs/jepa"
source "${VENV_PATH}/bin/activate"

REPO_DIR="${HOME}/JEPA-RobustViT"
cd "$REPO_DIR"
git pull origin main
mkdir -p logs checkpoints results

METHOD=${METHOD:-supervised}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}

echo "Method : $METHOD"
echo "Seed   : $SEED"
echo ""

# ── Step 1: Train linear probe ─────────────────────────────────────────────
if [ "$METHOD" = "supervised" ]; then
    echo "=== Step 1: Supervised linear probe ==="
    python scripts/train_linear_probe.py \
        --dataset    pathmnist     \
        --seed       "$SEED"       \
        --epochs     10            \
        --batch_size "$BATCH_SIZE" \
        --lr         1e-3          \
        --checkpoint_dir checkpoints

    CHECKPOINT="checkpoints/pathmnist_seed${SEED}.pth"

elif [ "$METHOD" = "dino" ] || [ "$METHOD" = "mae" ]; then
    echo "=== Step 1: ${METHOD^^} baseline linear probe ==="
    python scripts/eval_baselines.py \
        --method     "$METHOD"    \
        --seed       "$SEED"      \
        --epochs     10           \
        --batch_size "$BATCH_SIZE" \
        --lr         1e-3         \
        --checkpoint_dir checkpoints

    CHECKPOINT="checkpoints/${METHOD}_pathmnist_seed${SEED}.pth"

elif [ "$METHOD" = "jepa" ]; then
    echo "=== Step 1: JEPA linear probe ==="
    python scripts/train_linear_probe.py \
        --dataset    pathmnist     \
        --seed       "$SEED"       \
        --epochs     10            \
        --batch_size "$BATCH_SIZE" \
        --lr         1e-3          \
        --checkpoint_dir checkpoints

    CHECKPOINT="checkpoints/pathmnist_seed${SEED}.pth"
fi

echo ""
echo "=== Step 2: Domain shift evaluation ==="
python scripts/eval_domain_shift.py \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=== Step 3: TTA evaluation (1 step) ==="
python scripts/eval_tta.py \
    --checkpoint "$CHECKPOINT" \
    --method     "$METHOD"     \
    --seed       "$SEED"       \
    --batch_size "$BATCH_SIZE" \
    --tta_lr     1e-4          \
    --tta_steps  1             \
    --episodic

echo ""
echo "=== Step 4: TTA evaluation (3 steps) ==="
python scripts/eval_tta.py \
    --checkpoint "$CHECKPOINT" \
    --method     "$METHOD"     \
    --seed       "$SEED"       \
    --batch_size "$BATCH_SIZE" \
    --tta_lr     1e-4          \
    --tta_steps  3             \
    --episodic

echo ""
echo "All evaluation complete: $(date)"