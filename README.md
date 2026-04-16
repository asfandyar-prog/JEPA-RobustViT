<div align="center">

<h1>JEPA-RobustViT</h1>

<p><em>Predictive Self-Supervised Vision Transformers under Test-Time Distribution Shifts with Lightweight Test-Time Adaptation</em></p>

<p><strong>BSc Computer Science Thesis &nbsp;·&nbsp; Asfand Yar &nbsp;·&nbsp; University of Debrecen &nbsp;·&nbsp; 2026</strong></p>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-ViT--B/16-7C3AED?style=for-the-badge)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-10b981?style=for-the-badge)](https://github.com/asfandyar-prog/JEPA-RobustViT)

**[Live Results Dashboard](https://asfandyar-prog.github.io/JEPA-RobustViT/)**

</div>

---

## What is JEPA-RobustViT?

**JEPA-RobustViT** is a research framework that combines *predictive self-supervised pretraining* with *lightweight test-time adaptation* to produce Vision Transformers that remain robust under real-world distribution shifts.

The core insight: a ViT pretrained to **predict abstract representations** of masked image regions (rather than reconstructing pixels) learns deeper semantic features that transfer better across domains. When combined with entropy-based test-time adaptation — compatible with LayerNorm, requiring zero labels — the resulting model adapts automatically to unseen distributions at inference time.

**Real-world motivation:** A medical AI trained on tissue pathology images degrades catastrophically when deployed on skin lesion or retinal fundus data. Our supervised ViT baseline retains only **6.6% of source accuracy** on DermaMNIST under zero-shot transfer, with ECE rising from 0.014 to 0.889. This is the problem we solve.

---

## Architecture

The full pipeline has four stages:

**① Backbone (ViT-B/16)** — A Vision Transformer that splits a 224×224 image into 196 patches of 16×16 pixels. Each patch is projected to a 768-dimensional embedding. A learnable CLS token aggregates global context across all patches through 12 layers of multi-head self-attention. The CLS token output is the image representation.

**② I-JEPA Pretraining** — The backbone is pretrained using a Joint-Embedding Predictive Architecture. A context encoder processes visible patches; a target encoder (updated via exponential moving average, no backprop) processes the full image. A lightweight predictor is trained to predict target encoder representations of masked regions from context encoder outputs using L2 loss in representation space — never at the pixel level.

**③ Linear Probe** — The pretrained backbone is frozen. A single linear layer `nn.Linear(768, num_classes)` is trained on top. This isolates the quality of learned representations from the classifier capacity.

**④ Test-Time Adaptation** — At inference, entropy of predictions is minimized by updating only the LayerNorm scale (γ) and shift (β) parameters. No labels required. Compatible with ViT architecture. Runs on a single forward pass.

```
Input Image (224×224)
        │
        ▼
┌─────────────────────┐
│   Patch Embedding   │  196 patches × 768-dim + CLS token
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ViT-B/16 Encoder   │  12 × [LayerNorm → MHSA → LayerNorm → FFN]
│  (Frozen after SSL) │  Output: CLS token ∈ ℝ^768
└─────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────────────┐
│Linear│  │ TTA Module   │
│Probe │  │ Entropy min. │
│head  │  │ on LN params │
└──────┘  └──────────────┘
```

---

## Results

> Full results with visualizations: **[Live Dashboard](https://asfandyar-prog.github.io/JEPA-RobustViT/)**

### Supervised ViT Baseline (Phase 1 — Complete)

| Condition | Accuracy | ECE | Retained |
|-----------|----------|-----|---------|
| PathMNIST (source) | **80.90 ± 0.17%** | 0.0138 | 100% |
| → DermaMNIST | 5.31 ± 0.15% | 0.8891 | 6.6% |
| → BloodMNIST | 17.78 ± 0.24% | 0.7489 | 21.9% |
| → RetinaMNIST | 10.58 ± 0.29% | 0.7301 | 13.1% |

### Full Comparison Table (In Progress)

| Method | PathMNIST | →Derma | →Blood | →Retina |
|--------|-----------|--------|--------|---------|
| Supervised ViT | 80.90% | 5.31% | 17.78% | 10.58% |
| DINO ViT-B/16 | — | — | — | — |
| MAE ViT-B/16 | — | — | — | — |
| I-JEPA (ours) | — | — | — | — |
| Supervised + TTA | — | — | — | — |
| DINO + TTA | — | — | — | — |
| MAE + TTA | — | — | — | — |
| **I-JEPA + TTA (proposed)** | — | — | — | — |

---

## Repository Structure

```
JEPA-RobustViT/
├── src/
│   ├── models/
│   │   ├── ijepa_backbone.py     # ViT-B/16 feature extractor (timm)
│   │   ├── classifier.py         # Frozen backbone + linear head
│   │   ├── jepa_predictor.py     # JEPA predictor head (planned)
│   │   └── tta.py                # Entropy-based TTA (planned)
│   ├── data/
│   │   ├── medmnist_loader.py    # PathMNIST, DermaMNIST, BloodMNIST, RetinaMNIST
│   │   └── cifar_loader.py       # CIFAR-10 loader
│   └── utils/
│       ├── metrics.py            # Accuracy, ECE, AverageMeter
│       └── transforms.py        # Standardised preprocessing
├── scripts/
│   ├── train_linear_probe.py     # Baseline training (multi-seed, argparse)
│   ├── eval_domain_shift.py      # Domain shift evaluation
│   ├── eval_baselines.py         # DINO/MAE comparison (planned)
│   └── train_jepa.py             # I-JEPA pretraining (planned)
├── results/
│   ├── baselines.md              # Experiment results (markdown)
│   ├── baselines.html            # Interactive results dashboard
│   └── *.json / *.txt            # Raw result files
├── notebooks/
│   └── kaggle_train.ipynb        # Kaggle training notebook
├── docs/
│   └── index.html                # GitHub Pages dashboard
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart

```bash
git clone https://github.com/asfandyar-prog/JEPA-RobustViT.git
cd JEPA-RobustViT
pip install -r requirements.txt
```

**Train linear probe (3 seeds):**
```bash
python scripts/train_linear_probe.py --dataset pathmnist --seed 0 --epochs 7
python scripts/train_linear_probe.py --dataset pathmnist --seed 1 --epochs 7
python scripts/train_linear_probe.py --dataset pathmnist --seed 2 --epochs 7
```

**Evaluate domain shift:**
```bash
python scripts/eval_domain_shift.py --checkpoint checkpoints/pathmnist_seed0.pth
```

---

## Key Design Decisions

**Why I-JEPA over MAE?**
MAE reconstructs pixels — forcing the model to learn high-frequency texture details irrelevant to semantic understanding. I-JEPA predicts representations in embedding space, learning what things *are* rather than what they look like at pixel level. This produces more transferable features under domain shift.

**Why not TENT for TTA?**
TENT adapts BatchNorm statistics at test time. Vision Transformers use LayerNorm, not BatchNorm. Our entropy minimization approach directly optimizes the LayerNorm affine parameters (γ, β), making it natively compatible with ViT architectures.

**Why freeze the backbone during linear probe?**
A linear probe measures representation quality directly. If the backbone is fine-tuned, improved accuracy could come from adaptation rather than from better pretraining. Freezing isolates the SSL method's contribution.

---

## Theoretical Background

This work sits at the intersection of three research threads:

**Joint-Embedding Predictive Architectures** (Assran et al., CVPR 2023) — predict abstract representations of masked regions rather than raw pixels. The target encoder is updated via exponential moving average, avoiding representation collapse without contrastive pairs.

**Vision Transformers under Distribution Shift** — ViTs exhibit different robustness profiles than CNNs. Their lack of CNN inductive biases (locality, translation equivariance) makes them both more sensitive to shift and a cleaner testbed for studying self-supervised objectives independently of architecture priors.

**Test-Time Adaptation** — TENT (Wang et al., ICLR 2021) demonstrated that adapting normalization layers at inference significantly closes the clean→corrupted accuracy gap without labeled test data. Our work extends this to LayerNorm-based architectures.

---

## Supervisors

| Role | Name | Institution |
|------|------|-------------|
| Primary supervisor | Dr. Bogacsovics Gergő | University of Debrecen |
| External supervisor | Sergio Correa | BMW Group |

---

<div align="center">

**Asfand Yar** · BSc Computer Science · University of Debrecen, Hungary

[![GitHub](https://img.shields.io/badge/GitHub-asfandyar--prog-181717?style=flat-square&logo=github)](https://github.com/asfandyar-prog)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Asfand%20Yar-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/asfand-yar-3966b8291)
[![Email](https://img.shields.io/badge/Email-yarasfand886%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:yarasfand886@gmail.com)

</div>