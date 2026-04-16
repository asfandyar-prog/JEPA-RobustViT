# Baseline Results — JEPA-RobustViT

**Thesis:** Predictive Self-Supervised Vision Transformers under Test-Time Distribution Shifts  
**Author:** Asfand Yar | University of Debrecen  
**Supervisors:** Dr. Bogacsovics Gergő (Unideb) · Sergio Correa (BMW)  
**Last updated:** April 2026

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Backbone | ViT-B/16 (`vit_base_patch16_224`) |
| Pretraining | ImageNet supervised (timm) |
| Backbone status | Frozen during all linear probe training |
| Classifier head | `nn.Linear(768, num_classes)` |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | CosineAnnealingLR (T_max=7, η_min=1e-6) |
| Epochs | 7 |
| Batch size | 256 |
| Seeds | 0, 1, 2 |
| Image size | 224 × 224 |
| Normalization | ImageNet mean/std |
| Hardware | NVIDIA Tesla T4 (Kaggle) |

---

## 1. PathMNIST Linear Probe (Source Domain Baseline)

**Dataset:** PathMNIST — 9-class colon pathology tissue classification  
**Train split:** 89,996 images | **Val split:** 10,004 images | **Test split:** 7,180 images

### Per-Seed Results

| Seed | Best Val Acc | Test Accuracy | Test ECE |
|------|-------------|---------------|----------|
| 0 | 84.66% | 81.14% | 0.0128 |
| 1 | 84.51% | 80.78% | 0.0110 |
| 2 | 84.30% | 80.77% | 0.0175 |

### Summary

| Metric | Mean ± Std |
|--------|-----------|
| Test Accuracy | **80.90% ± 0.17%** |
| Test ECE | **0.0138 ± 0.0033** |

### Training Curves (Loss per Epoch)

| Epoch | Seed 0 Loss | Seed 1 Loss | Seed 2 Loss |
|-------|------------|------------|------------|
| 1 | 0.5758 | 0.5944 | 0.5889 |
| 2 | 0.3946 | 0.3943 | 0.3954 |
| 3 | 0.3591 | 0.3567 | 0.3566 |
| 4 | 0.3411 | 0.3398 | 0.3383 |
| 5 | 0.3283 | 0.3282 | 0.3293 |
| 6 | 0.3206 | 0.3217 | 0.3213 |
| 7 | 0.3185 | 0.3203 | 0.3198 |

**Observation:** Loss converges smoothly across all seeds. Minimal variance between seeds confirms stable training.

---

## 2. Domain Shift Evaluation

**Setup:** Model trained on PathMNIST (source) evaluated zero-shot on three target domains.  
**Purpose:** Quantify the severity of distribution shift as motivation for TTA.

### Shift Pair 1: PathMNIST → DermaMNIST
**Shift type:** Colon tissue → Skin lesion (7 classes)

| Seed | Source Acc | Target Acc | Abs Drop | Rel% | Source ECE | Target ECE |
|------|-----------|-----------|---------|------|-----------|-----------|
| 0 | 81.14% | 5.14% | -75.99% | 6.3% | 0.0128 | 0.8994 |
| 1 | 80.78% | 5.34% | -75.44% | 6.6% | 0.0110 | 0.8907 |
| 2 | 80.77% | 5.44% | -75.33% | 6.7% | 0.0175 | 0.8773 |
| **Mean** | **80.90%** | **5.31%** | **-75.59%** | **6.6%** | **0.0138** | **0.8891** |

### Shift Pair 2: PathMNIST → BloodMNIST
**Shift type:** Colon tissue → Blood cell type (8 classes)

| Seed | Source Acc | Target Acc | Abs Drop | Rel% | Source ECE | Target ECE |
|------|-----------|-----------|---------|------|-----------|-----------|
| 0 | 81.14% | 18.01% | -63.14% | 22.2% | 0.0128 | 0.7644 |
| 1 | 80.78% | 17.80% | -62.98% | 22.0% | 0.0110 | 0.7432 |
| 2 | 80.77% | 17.54% | -63.23% | 21.7% | 0.0175 | 0.7392 |
| **Mean** | **80.90%** | **17.78%** | **-63.12%** | **21.9%** | **0.0138** | **0.7489** |

### Shift Pair 3: PathMNIST → RetinaMNIST
**Shift type:** Colon tissue → Retinal fundus grading (5 classes)

| Seed | Source Acc | Target Acc | Abs Drop | Rel% | Source ECE | Target ECE |
|------|-----------|-----------|---------|------|-----------|-----------|
| 0 | 81.14% | 10.25% | -70.89% | 12.6% | 0.0128 | 0.7390 |
| 1 | 80.78% | 10.75% | -70.03% | 13.3% | 0.0110 | 0.7132 |
| 2 | 80.77% | 10.75% | -70.02% | 13.3% | 0.0175 | 0.7382 |
| **Mean** | **80.90%** | **10.58%** | **-70.31%** | **13.1%** | **0.0138** | **0.7301** |

---

## 3. Summary Table (Supervised ViT Baseline)

| Condition | Accuracy | ECE | Abs Drop | Rel% |
|-----------|----------|-----|---------|------|
| PathMNIST (source) | **80.90 ± 0.17%** | 0.0138 | — | 100% |
| → DermaMNIST | **5.31 ± 0.15%** | 0.8891 | -75.59% | 6.6% |
| → BloodMNIST | **17.78 ± 0.24%** | 0.7489 | -63.12% | 21.9% |
| → RetinaMNIST | **10.58 ± 0.29%** | 0.7301 | -70.31% | 13.1% |

---

## 4. Key Observations

**Catastrophic accuracy degradation under shift.**
The supervised ViT retains only 6.6% of source performance on DermaMNIST,
demonstrating that ImageNet-pretrained features fail completely under
significant domain shift in medical imaging.

**ECE collapses under shift.**
Source ECE of 0.0138 increases to 0.8891 on DermaMNIST — a 64× increase.
The model becomes severely overconfident in wrong predictions, making
calibration a critical failure mode alongside accuracy.

**Shift severity varies by semantic distance.**
DermaMNIST (skin) represents the largest shift from colon tissue,
followed by RetinaMNIST (retina) and BloodMNIST (blood cells).
This ordering provides a natural difficulty axis for evaluating
robustness methods.

**Results are highly stable across seeds.**
Standard deviation of 0.17% on source accuracy and ≤0.29% on all
target domains confirms reproducibility of findings.

---

## 5. Planned Comparisons (To Be Added)

The following methods will be evaluated under identical conditions
and added to this document as experiments complete:

| Method | Type | Status |
|--------|------|--------|
| Supervised ViT (this baseline) | Supervised | ✅ Complete |
| DINO ViT-B/16 | Self-supervised (contrastive) | ⬜ Planned |
| MAE ViT-B/16 | Self-supervised (reconstruction) | ⬜ Planned |
| I-JEPA ViT-B/16 | Self-supervised (predictive) | ⬜ Planned |
| Supervised ViT + TTA | Supervised + adaptation | ⬜ Planned |
| DINO + TTA | SSL + adaptation | ⬜ Planned |
| MAE + TTA | SSL + adaptation | ⬜ Planned |
| I-JEPA + TTA (proposed) | SSL + adaptation | ⬜ Planned |

---

## 6. Checkpoints

| Checkpoint | Val Acc | Test Acc | Location |
|-----------|---------|---------|----------|
| `pathmnist_seed0.pth` | 84.66% | 81.14% | `checkpoints/` |
| `pathmnist_seed1.pth` | 84.51% | 80.78% | `checkpoints/` |
| `pathmnist_seed2.pth` | 84.30% | 80.77% | `checkpoints/` |

> Note: Checkpoints are not committed to GitHub due to file size.
> Store locally and on external storage.