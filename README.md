<div align="center">

<h1>🦾 JEPA-RobustViT</h1>

<p><em>Predictive Self-Supervised Vision Transformers under Test-Time Distribution Shifts</em></p>
<p><strong>BSc Computer Science Thesis &nbsp;·&nbsp; Asfand Yar &nbsp;·&nbsp; University of Debrecen &nbsp;·&nbsp; 2026</strong></p>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-ViT--B/16-7C3AED?style=for-the-badge)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<!-- Badges Row 2 -->
[![Thesis](https://img.shields.io/badge/BSc%20Thesis-Debrecen%202026-f97316?style=for-the-badge&logo=academia&logoColor=white)](https://github.com/asfandyar-prog/JEPA-RobustViT)
[![Status](https://img.shields.io/badge/Status-Active%20Research-10b981?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/asfandyar-prog/JEPA-RobustViT)
[![Author](https://img.shields.io/badge/Author-Asfand%20Yar-6366f1?style=for-the-badge&logo=github&logoColor=white)](https://github.com/asfandyar-prog)

</div>

---

## 🔍 What is JEPA-RobustViT?

**JEPA-RobustViT** is a research framework combining *predictive self-supervised learning* — inspired by Yann LeCun's Joint-Embedding Predictive Architecture (I-JEPA) — with **test-time adaptation (TTA)** to produce Vision Transformers that stay robust under real-world distribution shifts.

Standard supervised ViTs degrade significantly under domain shift. Our supervised ViT-B/16 baseline retains only **6.6% of source accuracy** on DermaMNIST under zero-shot transfer, with ECE rising from 0.014 to 0.889. This work investigates whether learning to **predict representations in embedding space** (not pixels) produces features inherently resistant to domain change — and whether lightweight TTA can close the remaining gap at inference time, without any labels.

> *"A model that can predict its own latent future is a model that understands its world."*

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    IMG["🖼️  Input Image\n224 × 224 × 3"]

    subgraph EMBED["① Patch Embedding"]
        direction LR
        P1["P₁"] --- P2["P₂"] --- P3["P₃"] --- P4["P₄"] --- P5["P₅"] --- P6["P₆"]
        CLS["[CLS]"]
    end

    subgraph BLOCK["② Transformer Encoder Block  ×12"]
        direction TB

        LN1["📐 Layer Norm 1\nx̂ = (x−μ)/σ · γ + β"]

        subgraph MHSA["Multi-Head Self-Attention  (h = 12 heads)"]
            direction LR
            Q["Q\nW_Q · x"] 
            K["K\nW_K · x"] 
            V["V\nW_V · x"]
            SDPA["Scaled Dot-Product\nsoftmax(QKᵀ/√d_k)·V"]
            CONCAT["Concat heads\n→ W_O"]
            Q --> SDPA
            K --> SDPA
            V --> SDPA
            SDPA --> CONCAT
        end

        ADD1(("⊕\nAdd"))
        LN2["📐 Layer Norm 2\nx̂ = (x−μ)/σ · γ + β"]

        subgraph FFN["Feed-Forward Network"]
            direction LR
            L1["Linear₁\n768 → 3072"] --> GELU["GELU\nσ(x)·x"] --> L2["Linear₂\n3072 → 768"]
        end

        ADD2(("⊕\nAdd"))

        LN1 --> MHSA
        MHSA --> ADD1
        ADD1 --> LN2
        LN2 --> FFN
        FFN --> ADD2
    end

    subgraph JEPA["③ I-JEPA Pretraining"]
        direction TB
        CTX["Context Encoder\nvisible patches → features"]
        TGT["Target Encoder\nEMA of context encoder"]
        PRED["Predictor\nf(z_context, positions)"]
        LOSS["L = ‖ẑ − z_target‖²\npredict in embedding space — no pixels"]
        CTX --> PRED
        TGT --> LOSS
        PRED --> LOSS
    end

    subgraph TTA["④ Test-Time Adaptation"]
        direction TB
        ENT["Entropy Min.\nH(p) = −Σ p·log(p) → min"]
        LN["LayerNorm Params\nupdate γ, β only"]
        NL["No labels required\nlightweight — runs at inference"]
        ENT --> LN --> NL
    end

    CLS --> EMBED
    EMBED --> BLOCK
    IMG --> EMBED

    BLOCK --> JEPA
    BLOCK --> TTA

    CLS2["🎯 Linear Classifier\nW ∈ ℝ^{768×C} · softmax\nRobust Prediction ✓"]

    JEPA --> CLS2
    TTA --> CLS2

    style IMG fill:#1e293b,stroke:#64748b,color:#e2e8f0
    style EMBED fill:#0f172a,stroke:#3b82f6,color:#93c5fd
    style BLOCK fill:#0f0a2a,stroke:#7c3aed,color:#c4b5fd
    style MHSA fill:#1e1b4b,stroke:#6366f1,color:#a5b4fc
    style FFN fill:#064e3b,stroke:#10b981,color:#6ee7b7
    style LN1 fill:#2e1065,stroke:#7c3aed,color:#c4b5fd
    style LN2 fill:#2e1065,stroke:#7c3aed,color:#c4b5fd
    style ADD1 fill:#083344,stroke:#06b6d4,color:#06b6d4
    style ADD2 fill:#1e0a40,stroke:#a855f7,color:#a855f7
    style Q fill:#1e3a8a,stroke:#3b82f6,color:#93c5fd
    style K fill:#14532d,stroke:#10b981,color:#6ee7b7
    style V fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
    style SDPA fill:#1e1b4b,stroke:#818cf8,color:#a5b4fc
    style CONCAT fill:#312e81,stroke:#6366f1,color:#a5b4fc
    style L1 fill:#064e3b,stroke:#059669,color:#a7f3d0
    style GELU fill:#065f46,stroke:#34d399,color:#6ee7b7
    style L2 fill:#064e3b,stroke:#059669,color:#a7f3d0
    style JEPA fill:#022c22,stroke:#059669,color:#34d399
    style CTX fill:#053d2a,stroke:#10b981,color:#6ee7b7
    style TGT fill:#053d2a,stroke:#10b981,color:#6ee7b7
    style PRED fill:#064e3b,stroke:#34d399,color:#a7f3d0
    style LOSS fill:#065f46,stroke:#6ee7b7,color:#d1fae5
    style TTA fill:#1a0803,stroke:#ea580c,color:#fb923c
    style ENT fill:#3b1108,stroke:#f97316,color:#fdba74
    style LN fill:#3b1108,stroke:#ea580c,color:#fdba74
    style NL fill:#3b1108,stroke:#c2410c,color:#fdba74
    style CLS2 fill:#1e0a40,stroke:#6366f1,color:#a5b4fc
```

---

## 📊 Research Contributions

| # | Contribution | Status |
|---|---|---|
| 🔵 | **Supervised ViT Baseline** — ViT-B/16, ImageNet pretrained, PathMNIST linear probe | ✅ Complete |
| 🔵 | **Domain Shift Evaluation** — PathMNIST → DermaMNIST / BloodMNIST / RetinaMNIST | ✅ Complete |
| 🟣 | **DINO Baseline** — official pretrained ViT-B/16, linear probe + shift eval | 🔄 In Progress |
| 🟣 | **MAE Baseline** — official pretrained ViT-B/16, linear probe + shift eval | 🔄 In Progress |
| 🟠 | **I-JEPA Pretraining** — context→target embedding prediction, EMA target encoder | 🔄 Planned |
| 🟠 | **TTA: Entropy Minimization** — LayerNorm-compatible, no labels, lightweight | 🔄 Planned |
| ⚪ | **Driving Domain** — distribution shift evaluation on autonomous driving data | 🔄 Planned |

---

## 📁 Repository Structure

```
JEPA-RobustViT/
│
├── 📂 src/
│   ├── 📂 models/
│   │   ├── ijepa_backbone.py    # ViT-B/16 feature extractor (timm)
│   │   ├── classifier.py        # Frozen backbone + linear head
│   │   ├── jepa_predictor.py    # JEPA predictor head (planned)
│   │   └── tta.py               # Entropy-based TTA (planned)
│   ├── 📂 data/
│   │   ├── medmnist_loader.py   # PathMNIST, DermaMNIST, BloodMNIST, RetinaMNIST
│   │   └── cifar_loader.py      # CIFAR-10 loader
│   └── 📂 utils/
│       ├── metrics.py           # Accuracy, ECE, AverageMeter
│       └── transforms.py        # Standardised preprocessing
│
├── 📂 scripts/
│   ├── train_linear_probe.py    # Baseline training (multi-seed, argparse)
│   ├── eval_domain_shift.py     # Domain shift evaluation
│   ├── eval_baselines.py        # DINO/MAE comparison (planned)
│   └── train_jepa.py            # I-JEPA pretraining (planned)
│
├── 📂 results/
│   ├── baselines.md             # Tracked experiment results
│   ├── baselines.html           # Interactive results dashboard
│   └── *.json / *.txt           # Raw result files per seed
│
├── 📂 notebooks/
│   └── kaggle_train.ipynb       # Kaggle training notebook
│
├── 📂 docs/
│   └── index.html               # GitHub Pages live dashboard
│
├── main.py                      # Entry point
├── pyproject.toml               # Project configuration
└── requirements.txt             # Dependencies
```

---

## 🚀 Quickstart

**Clone & install:**
```bash
git clone https://github.com/asfandyar-prog/JEPA-RobustViT.git
cd JEPA-RobustViT
pip install -r requirements.txt
```

**Or with `uv` (recommended):**
```bash
uv sync
```

**Train linear probe (3 seeds):**
```bash
python scripts/train_linear_probe.py --dataset pathmnist --seed 0 --epochs 7
python scripts/train_linear_probe.py --dataset pathmnist --seed 1 --epochs 7
python scripts/train_linear_probe.py --dataset pathmnist --seed 2 --epochs 7
```

**Run domain shift evaluation:**
```bash
python scripts/eval_domain_shift.py --checkpoint checkpoints/pathmnist_seed0.pth
```

---

## 🧠 Key Design Decisions

### Why I-JEPA over MAE?
Masked Autoencoders reconstruct pixels — a task requiring high-frequency detail but not semantic understanding. I-JEPA predicts **representations in embedding space**, forcing the model to reason at the level of meaning. The EMA-updated target encoder provides stable training targets without contrastive pairs. This is hypothesized to yield features more transferable under domain shift.

### Why not TENT for TTA?
TENT adapts BatchNorm statistics at test time. Vision Transformers use **LayerNorm, not BatchNorm** — TENT cannot be applied directly to ViTs. Our entropy minimization approach updates only the LayerNorm affine parameters (γ, β), making it natively compatible with ViT architectures without any structural changes.

### Why ViT?
Transformers lack the inductive biases (translation equivariance, locality) of CNNs. This makes them both more sensitive to distribution shift *and* a cleaner testbed for studying what self-supervised objectives contribute to robustness independently of architecture priors.

---

## 📈 Results

> Full interactive dashboard: **[asfandyar-prog.github.io/JEPA-RobustViT](https://asfandyar-prog.github.io/JEPA-RobustViT/)**  
> Full benchmark table: [`results/baselines.md`](results/baselines.md)

### Supervised ViT-B/16 Baseline (3 seeds)

| Condition | Accuracy | ECE | Retained |
|---|---|---|---|
| PathMNIST (source) | **80.90 ± 0.17%** | 0.0138 | 100% |
| → DermaMNIST (shift 1) | 5.31 ± 0.15% | 0.8891 | 6.6% |
| → BloodMNIST (shift 2) | 17.78 ± 0.24% | 0.7489 | 21.9% |
| → RetinaMNIST (shift 3) | 10.58 ± 0.29% | 0.7301 | 13.1% |

### Full Comparison Table (In Progress)

| Method | PathMNIST | →Derma | →Blood | →Retina |
|---|---|---|---|---|
| Supervised ViT-B/16 | 80.90% | 5.31% | 17.78% | 10.58% |
| DINO ViT-B/16 | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |
| MAE ViT-B/16 | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |
| **I-JEPA ViT-B/16 (ours)** | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |
| Supervised + TTA | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |
| DINO + TTA | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |
| MAE + TTA | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |
| **I-JEPA + TTA (proposed)** | 🔄 WIP | 🔄 WIP | 🔄 WIP | 🔄 WIP |

---

## 📚 Theoretical Background

This work sits at the intersection of three active research threads:

**1. Joint-Embedding Predictive Architectures (Assran et al., CVPR 2023)**
Predict abstract representations of masked regions rather than raw pixels — learning rich semantic features. The EMA-updated target encoder provides stable training targets without contrastive pairs or pixel reconstruction.

**2. Vision Transformers under Distribution Shift**
ViTs exhibit different robustness profiles than CNNs ([Bhojanapalli et al., 2021](https://arxiv.org/abs/2104.02821); [Paul & Chen, 2022](https://arxiv.org/abs/2105.07581)) — understanding *why* is a core question this thesis addresses.

**3. Test-Time Adaptation**
TENT ([Wang et al., 2021](https://arxiv.org/abs/2006.10726)) and subsequent TTA methods demonstrate that adapting normalization layers at inference significantly closes the clean→corrupted accuracy gap without labeled test data.

---

## ⚙️ Dependencies

```toml
[dependencies]
torch       = ">=2.0"
torchvision = ">=0.15"
timm        = ">=0.9"
einops      = ">=0.7"
numpy       = ">=1.24"
tqdm        = ">=4.65"
medmnist    = ">=2.0"
```

---

## 👤 About

<div align="center">

**Asfand Yar** · BSc Computer Science · University of Debrecen, Hungary

*Thesis project 2025–2026 · Co-supervised by Dr. Bogacsovics Gergő (Unideb) & Sergio Correa (BMW)*

[![GitHub](https://img.shields.io/badge/GitHub-asfandyar--prog-181717?style=flat-square&logo=github)](https://github.com/asfandyar-prog)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Asfand%20Yar-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/asfand-yar-3966b8291)
[![Email](https://img.shields.io/badge/Email-yarasfand886%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:yarasfand886@gmail.com)

<br/>

*If this work is useful to your research, a ⭐ helps the project grow.*

</div>