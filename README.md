<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=JEPA-RobustViT&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Predictive%20Self-Supervised%20Vision%20Transformers%20under%20Distribution%20Shift&descAlignY=58&descSize=15&animation=fadeIn" width="100%"/>

<!-- Animated Transformer Architecture SVG -->
<svg width="860" height="320" viewBox="0 0 860 320" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glow-blue" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-purple" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-cyan" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="bg-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a0a1a"/>
      <stop offset="100%" style="stop-color:#12082a"/>
    </linearGradient>
    <linearGradient id="patch-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1e3a5f"/>
      <stop offset="100%" style="stop-color:#0d2137"/>
    </linearGradient>
    <linearGradient id="attn-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#2d1b69"/>
      <stop offset="100%" style="stop-color:#1a0e3d"/>
    </linearGradient>
    <linearGradient id="jepa-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0e3d2d"/>
      <stop offset="100%" style="stop-color:#071f17"/>
    </linearGradient>
    <linearGradient id="tta-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#3d1a0e"/>
      <stop offset="100%" style="stop-color:#1f0e07"/>
    </linearGradient>
    <linearGradient id="arrow-flow" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#00d4ff;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#00d4ff;stop-opacity:0"/>
    </linearGradient>
    <linearGradient id="arrow-flow2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#a855f7;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#a855f7;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#a855f7;stop-opacity:0"/>
    </linearGradient>
    <linearGradient id="arrow-flow3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#10b981;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#10b981;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#10b981;stop-opacity:0"/>
    </linearGradient>
    <pattern id="dots" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
      <circle cx="1" cy="1" r="1" fill="#6366f1"/>
    </pattern>
  </defs>

  <!-- Background -->
  <rect width="860" height="320" fill="url(#bg-grad)" rx="12"/>
  <rect width="860" height="320" fill="url(#dots)" rx="12" opacity="0.06"/>

  <!-- ══ BLOCK 1: PATCH EMBEDDING ══ -->
  <rect x="20" y="80" width="120" height="120" rx="8" fill="url(#patch-grad)" stroke="#1e88e5" stroke-width="1.5" filter="url(#glow-blue)"/>
  <g stroke="#2a6496" stroke-width="0.8" opacity="0.6">
    <line x1="50" y1="80" x2="50" y2="200"/><line x1="80" y1="80" x2="80" y2="200"/><line x1="110" y1="80" x2="110" y2="200"/>
    <line x1="20" y1="110" x2="140" y2="110"/><line x1="20" y1="140" x2="140" y2="140"/><line x1="20" y1="170" x2="140" y2="170"/>
  </g>
  <rect x="20" y="80" width="30" height="30" fill="#1e88e5" opacity="0.25" rx="2"><animate attributeName="opacity" values="0.25;0.6;0.25" dur="2s" repeatCount="indefinite"/></rect>
  <rect x="50" y="110" width="30" height="30" fill="#1e88e5" opacity="0.2" rx="2"><animate attributeName="opacity" values="0.2;0.55;0.2" dur="2s" begin="0.3s" repeatCount="indefinite"/></rect>
  <rect x="80" y="140" width="30" height="30" fill="#1e88e5" opacity="0.3" rx="2"><animate attributeName="opacity" values="0.3;0.65;0.3" dur="2s" begin="0.6s" repeatCount="indefinite"/></rect>
  <rect x="110" y="110" width="30" height="30" fill="#42a5f5" opacity="0.2" rx="2"><animate attributeName="opacity" values="0.2;0.5;0.2" dur="2.4s" begin="0.9s" repeatCount="indefinite"/></rect>
  <line x1="20" y1="100" x2="140" y2="100" stroke="#00d4ff" stroke-width="1.5" opacity="0.7">
    <animate attributeName="y1" values="80;200;80" dur="3s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="80;200;80" dur="3s" repeatCount="indefinite"/>
  </line>
  <text x="80" y="218" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#64b5f6" font-weight="bold">PATCH EMBED</text>
  <text x="80" y="232" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#4a7fa8">16×16 patches</text>

  <!-- Arrow 1 -->
  <line x1="145" y1="140" x2="185" y2="140" stroke="#1a3a4a" stroke-width="2"/>
  <rect x="145" y="136" width="40" height="8" fill="url(#arrow-flow)" rx="4"><animate attributeName="x" values="145;185;145" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;1;0" dur="2s" repeatCount="indefinite"/></rect>
  <polygon points="183,135 191,140 183,145" fill="#00d4ff" filter="url(#glow-cyan)"/>

  <!-- ══ BLOCK 2: MULTI-HEAD ATTENTION ══ -->
  <rect x="190" y="60" width="160" height="200" rx="10" fill="url(#attn-grad)" stroke="#7c3aed" stroke-width="1.5" filter="url(#glow-purple)"/>
  <text x="270" y="83" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#a78bfa" font-weight="bold">MULTI-HEAD</text>
  <text x="270" y="95" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#a78bfa" font-weight="bold">ATTENTION</text>
  <g font-family="'Courier New', monospace" font-size="9" fill="#c4b5fd">
    <text x="215" y="118">Q</text><text x="263" y="118">K</text><text x="311" y="118">V</text>
  </g>
  <rect x="208" y="122" width="26" height="16" rx="3" fill="#4c1d95" stroke="#7c3aed" stroke-width="1"/>
  <rect x="256" y="122" width="26" height="16" rx="3" fill="#4c1d95" stroke="#7c3aed" stroke-width="1"/>
  <rect x="304" y="122" width="26" height="16" rx="3" fill="#4c1d95" stroke="#7c3aed" stroke-width="1"/>
  <ellipse cx="225" cy="175" rx="20" ry="12" fill="none" stroke="#818cf8" stroke-width="1.2"><animate attributeName="rx" values="20;25;20" dur="2.5s" repeatCount="indefinite"/></ellipse>
  <ellipse cx="270" cy="170" rx="20" ry="14" fill="none" stroke="#a855f7" stroke-width="1.2"><animate attributeName="rx" values="20;24;20" dur="2.5s" begin="0.4s" repeatCount="indefinite"/></ellipse>
  <ellipse cx="315" cy="175" rx="20" ry="12" fill="none" stroke="#c084fc" stroke-width="1.2"><animate attributeName="rx" values="20;25;20" dur="2.5s" begin="0.8s" repeatCount="indefinite"/></ellipse>
  <path d="M215,163 Q270,148 315,163" stroke="#6366f1" stroke-width="1" fill="none" opacity="0.6"><animate attributeName="d" values="M215,163 Q270,148 315,163;M215,163 Q270,145 315,163;M215,163 Q270,148 315,163" dur="3s" repeatCount="indefinite"/></path>
  <path d="M220,168 Q270,158 318,168" stroke="#8b5cf6" stroke-width="0.8" fill="none"><animate attributeName="opacity" values="0.4;0.8;0.4" dur="2s" repeatCount="indefinite"/></path>
  <text x="270" y="210" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#818cf8">softmax(QKᵀ/√d)V</text>
  <rect x="215" y="218" width="110" height="18" rx="4" fill="#2e1065" stroke="#7c3aed" stroke-width="0.8"/>
  <text x="270" y="231" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#c4b5fd">FFN + LayerNorm</text>

  <!-- Arrow 2 -->
  <line x1="355" y1="140" x2="395" y2="140" stroke="#2a1a4a" stroke-width="2"/>
  <rect x="355" y="136" width="40" height="8" fill="url(#arrow-flow2)" rx="4"><animate attributeName="x" values="355;395;355" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;1;0" dur="2s" repeatCount="indefinite"/></rect>
  <polygon points="393,135 401,140 393,145" fill="#a855f7" filter="url(#glow-purple)"/>

  <!-- ══ BLOCK 3: JEPA PREDICTOR ══ -->
  <rect x="400" y="70" width="140" height="185" rx="10" fill="url(#jepa-grad)" stroke="#10b981" stroke-width="1.5" filter="url(#glow-blue)"/>
  <text x="470" y="92" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#34d399" font-weight="bold">JEPA</text>
  <text x="470" y="104" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#34d399" font-weight="bold">PREDICTOR</text>
  <rect x="415" y="115" width="110" height="26" rx="5" fill="#064e3b" stroke="#10b981" stroke-width="1"/>
  <text x="470" y="132" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#6ee7b7">Context Encoder</text>
  <polygon points="466,141 470,149 474,141" fill="#10b981"/>
  <rect x="415" y="152" width="110" height="26" rx="5" fill="#065f46" stroke="#34d399" stroke-width="1"><animate attributeName="stroke-opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite"/></rect>
  <text x="470" y="169" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#a7f3d0">Target Prediction</text>
  <g opacity="0.7">
    <rect x="418" y="186" width="14" height="14" rx="2" fill="#10b981" opacity="0.6"><animate attributeName="opacity" values="0.6;0.2;0.6" dur="1.5s" repeatCount="indefinite"/></rect>
    <rect x="436" y="186" width="14" height="14" rx="2" fill="none" stroke="#10b981" stroke-dasharray="3,2"/>
    <rect x="454" y="186" width="14" height="14" rx="2" fill="none" stroke="#10b981" stroke-dasharray="3,2"/>
    <rect x="472" y="186" width="14" height="14" rx="2" fill="#10b981" opacity="0.5"><animate attributeName="opacity" values="0.5;0.15;0.5" dur="1.5s" begin="0.5s" repeatCount="indefinite"/></rect>
    <rect x="490" y="186" width="14" height="14" rx="2" fill="none" stroke="#10b981" stroke-dasharray="3,2"/>
    <rect x="508" y="186" width="14" height="14" rx="2" fill="#10b981" opacity="0.45"><animate attributeName="opacity" values="0.45;0.1;0.45" dur="1.5s" begin="1s" repeatCount="indefinite"/></rect>
  </g>
  <text x="470" y="215" text-anchor="middle" font-family="'Courier New', monospace" font-size="7.5" fill="#6ee7b7">masked prediction</text>
  <text x="470" y="245" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#34d399">embedding space</text>

  <!-- Arrow 3 -->
  <line x1="545" y1="140" x2="585" y2="140" stroke="#1a2a1a" stroke-width="2"/>
  <rect x="545" y="136" width="40" height="8" fill="url(#arrow-flow3)" rx="4"><animate attributeName="x" values="545;585;545" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;1;0" dur="2s" repeatCount="indefinite"/></rect>
  <polygon points="583,135 591,140 583,145" fill="#10b981"/>

  <!-- ══ BLOCK 4: TTA ══ -->
  <rect x="595" y="70" width="140" height="185" rx="10" fill="url(#tta-grad)" stroke="#f97316" stroke-width="1.5"/>
  <text x="665" y="92" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#fb923c" font-weight="bold">TEST-TIME</text>
  <text x="665" y="104" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#fb923c" font-weight="bold">ADAPTATION</text>
  <rect x="610" y="115" width="110" height="22" rx="4" fill="#431407" stroke="#f97316" stroke-width="0.8"/>
  <text x="665" y="130" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#fed7aa">Entropy Minimization</text>
  <rect x="610" y="143" width="110" height="22" rx="4" fill="#431407" stroke="#ea580c" stroke-width="0.8"/>
  <text x="665" y="158" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#fed7aa">BatchNorm Adapt</text>
  <rect x="610" y="171" width="110" height="22" rx="4" fill="#431407" stroke="#c2410c" stroke-width="0.8"/>
  <text x="665" y="186" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#fed7aa">Distribution Shift</text>
  <polyline points="612,215 620,208 628,222 636,205 644,218 652,210 660,224 668,207 676,220 684,212 692,218 700,208" stroke="#f97316" stroke-width="1.5" fill="none" opacity="0.7">
    <animate attributeName="points" values="612,215 620,208 628,222 636,205 644,218 652,210 660,224 668,207 676,220 684,212 692,218 700,208;612,210 620,222 628,208 636,220 644,207 652,218 660,210 668,222 676,208 684,220 692,210 700,215;612,215 620,208 628,222 636,205 644,218 652,210 660,224 668,207 676,220 684,212 692,218 700,208" dur="2s" repeatCount="indefinite"/>
  </polyline>
  <text x="665" y="245" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#fb923c">ImageNet-C/R eval</text>

  <!-- Final arrow -->
  <line x1="738" y1="140" x2="770" y2="140" stroke="#2a1a0a" stroke-width="2"/>
  <rect x="738" y="136" width="32" height="8" fill="#f97316" rx="4" opacity="0.7"><animate attributeName="opacity" values="0.7;0.2;0.7" dur="1.8s" repeatCount="indefinite"/></rect>
  <polygon points="768,135 776,140 768,145" fill="#f97316"/>

  <!-- ══ OUTPUT ══ -->
  <rect x="778" y="112" width="68" height="58" rx="8" fill="#0f0f1f" stroke="#6366f1" stroke-width="1.5" filter="url(#glow-purple)"/>
  <text x="812" y="133" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#818cf8">Robust</text>
  <text x="812" y="145" text-anchor="middle" font-family="'Courier New', monospace" font-size="8" fill="#818cf8">Predict</text>
  <rect x="790" y="152" width="44" height="5" rx="2" fill="#1e1e3f"/>
  <rect x="790" y="152" width="38" height="5" rx="2" fill="#6366f1"><animate attributeName="width" values="38;44;38" dur="3s" repeatCount="indefinite"/></rect>
  <rect x="790" y="160" width="44" height="5" rx="2" fill="#1e1e3f"/>
  <rect x="790" y="160" width="24" height="5" rx="2" fill="#a855f7"><animate attributeName="width" values="24;30;24" dur="2.5s" begin="0.5s" repeatCount="indefinite"/></rect>

  <!-- Floating particles -->
  <g filter="url(#glow-cyan)" opacity="0.6">
    <circle r="2" fill="#00d4ff"><animateMotion dur="8s" repeatCount="indefinite" path="M 155,140 C 200,100 300,180 390,140 C 480,100 560,175 590,140"/><animate attributeName="opacity" values="0;1;1;0" dur="8s" repeatCount="indefinite"/></circle>
    <circle r="1.5" fill="#a855f7"><animateMotion dur="8s" begin="2.5s" repeatCount="indefinite" path="M 155,145 C 200,170 300,120 390,145 C 480,170 560,130 590,145"/><animate attributeName="opacity" values="0;1;1;0" dur="8s" begin="2.5s" repeatCount="indefinite"/></circle>
    <circle r="1.5" fill="#10b981"><animateMotion dur="4s" begin="1s" repeatCount="indefinite" path="M 545,140 C 560,120 580,160 590,140"/><animate attributeName="opacity" values="0;1;1;0" dur="4s" begin="1s" repeatCount="indefinite"/></circle>
  </g>

  <!-- Stage labels -->
  <text x="80" y="285" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#4a7fa8">① Vision</text>
  <text x="270" y="285" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#7c3aed">② Encoding</text>
  <text x="470" y="285" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#10b981">③ Prediction</text>
  <text x="665" y="285" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#f97316">④ Adaptation</text>
  <text x="812" y="285" text-anchor="middle" font-family="'Courier New', monospace" font-size="9" fill="#6366f1">⑤ Output</text>
</svg>

<br/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-ViT%20Backbone-blueviolet?style=for-the-badge)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<!-- Badges Row 2 -->
[![Thesis](https://img.shields.io/badge/BSc%20Thesis-Debrecen%202026-f97316?style=for-the-badge&logo=academia&logoColor=white)](https://github.com/asfandyar-prog/JEPA-RobustViT)
[![Status](https://img.shields.io/badge/Status-Active%20Research-10b981?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/asfandyar-prog/JEPA-RobustViT)
[![Author](https://img.shields.io/badge/Author-Asfand%20Yar-6366f1?style=for-the-badge&logo=github&logoColor=white)](https://github.com/asfandyar-prog)

</div>

---

## ✦ What is JEPA-RobustViT?

**JEPA-RobustViT** is a research framework exploring how *predictive self-supervised objectives* — inspired by Yann LeCun's Joint-Embedding Predictive Architecture (I-JEPA) — can be combined with **test-time adaptation (TTA)** to produce Vision Transformers that remain robust under real-world distribution shifts.

Standard supervised ViTs degrade significantly when evaluated on corrupted or out-of-distribution inputs. This work investigates whether learning to **predict representations in embedding space** (rather than pixels) produces features inherently more resistant to shifts like noise, blur, weather corruptions, and domain change — and whether TTA techniques can further close the generalization gap at inference time with no labels required.

> *"A model that can predict its own latent future is a model that understands its world."*

---

## ✦ Architecture Overview

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PATCH EMBEDDING                              │
│   16×16 patches → linear projection → positional encoding      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                ViT BACKBONE  (timm / ViT-S/16)                  │
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│   │  Multi-Head  │   │  Multi-Head  │   │  Multi-Head  │       │
│   │  Self-Attn   │ → │  Self-Attn   │ → │  Self-Attn   │  ×L  │
│   └──────────────┘   └──────────────┘   └──────────────┘       │
│        [CLS] token + patch tokens → rich representations        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
┌──────────────────────┐    ┌────────────────────────────────────┐
│   JEPA PREDICTOR     │    │     TEST-TIME ADAPTATION           │
│                      │    │                                    │
│  Context Patches ──► │    │  • Entropy Minimization            │
│  Predict Target      │    │  • BatchNorm Statistics Update     │
│  Embeddings          │    │  • Augmentation Consistency        │
│  (masked regions)    │    │  • No labels required              │
└──────────────────────┘    └────────────────────────────────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                    ┌───────────────┐
                    │  CLASSIFIER   │
                    │  (linear prob)│
                    └───────────────┘
                            │
                   Robust Prediction ✓
```

---

## ✦ Research Contributions

| # | Contribution | Status |
|---|---|---|
| 🔵 | **ViT Baseline** — timm ViT-S/16, pretrained on ImageNet-1K | ✅ Complete |
| 🟣 | **JEPA Predictive Head** — context→target embedding prediction, masked patch strategy | ✅ Complete |
| 🟢 | **TTA: Entropy Minimization** — minimizes prediction entropy at test time | ✅ Complete |
| 🟢 | **TTA: BatchNorm Adaptation** — adapts running statistics to test distribution | ✅ Complete |
| 🟠 | **ImageNet-C Evaluation** — 15 corruption types × 5 severity levels | 🔄 In Progress |
| 🟠 | **ImageNet-R Evaluation** — rendition/style domain shift benchmark | 🔄 In Progress |
| ⚪ | **Medical Domain Transfer** — ChestX-ray14, DermaMNIST loaders | 🔄 In Progress |

---

## ✦ Repository Structure

```
JEPA-RobustViT/
│
├── 📂 src/
│   ├── backbone.py          # ViT encoder (timm wrapper)
│   ├── predictor.py         # JEPA predictive head
│   ├── classifier.py        # Linear probe classifier
│   └── tta.py               # Test-time adaptation modules
│
├── 📂 scripts/
│   ├── train_backbone.py    # Backbone pretraining
│   ├── train_jepa.py        # JEPA objective training
│   ├── test_backbone.py     # Evaluation pipeline
│   └── eval_robustness.py   # ImageNet-C/R benchmarks
│
├── 📂 results/
│   └── baselines.md         # Tracked experiment results
│
├── main.py                  # Entry point
├── pyproject.toml           # Project configuration
└── requirements.txt         # Dependencies
```

---

## ✦ Quickstart

**Clone & install:**
```bash
git clone https://github.com/asfandyar-prog/JEPA-RobustViT.git
cd JEPA-RobustViT
pip install -r requirements.txt
```

**Or with `uv` (recommended — respects `.python-version`):**
```bash
uv sync
```

**Run baseline evaluation:**
```bash
python scripts/test_backbone.py --model vit_small_patch16_224 --dataset cifar10
```

**Train with JEPA objective:**
```bash
python main.py --mode jepa --backbone vit_small_patch16_224 --epochs 100
```

**Run TTA robustness evaluation:**
```bash
python scripts/eval_robustness.py --tta entropy --dataset imagenet-c --severity 3
```

---

## ✦ Key Design Decisions

### Why JEPA over MAE?
Masked Autoencoders (MAE) reconstruct pixels — a task that requires modeling high-frequency detail but not necessarily semantic understanding. JEPA predicts **representations**, forcing the model to reason at the level of meaning. This is hypothesized to yield features more invariant to low-level corruptions like noise and blur.

### Why Test-Time Adaptation?
Even robust pretraining cannot cover all distribution shifts encountered at deployment. TTA allows the model to adapt its normalization statistics and reduce prediction entropy on a specific test batch — with **no labels required**, making it practical for real-world deployment.

### Why ViT?
Transformers lack the inductive biases (translation equivariance, locality) of CNNs. This makes them both more sensitive to distribution shift *and* a cleaner testbed for studying what self-supervised objectives contribute to robustness — independently of architectural priors.

---

## ✦ Results (Preliminary)

> Full benchmark table in [`results/baselines.md`](results/baselines.md)

| Model | ImageNet-1K (clean) | ImageNet-C (mCE ↓) | Notes |
|---|---|---|---|
| ViT-S/16 Supervised | ~79.8% | ~55.4 | Baseline |
| ViT-S/16 + MAE | ~83.1% | ~49.2 | Pixel reconstruction |
| **ViT-S/16 + JEPA** *(ours)* | 🔄 WIP | 🔄 WIP | Embedding prediction |
| **+ TTA Entropy Min** *(ours)* | 🔄 WIP | 🔄 WIP | Inference-time adapt |

---

## ✦ Theoretical Background

This work sits at the intersection of three active research threads:

**1. Joint-Embedding Predictive Architectures (LeCun, 2022)**
Predict abstract representations of masked/future inputs rather than raw pixels — learning rich semantic features without generative pixel reconstruction.

**2. Vision Transformers under Distribution Shift**
ViTs exhibit different robustness profiles than CNNs ([Bhojanapalli et al., 2021](https://arxiv.org/abs/2104.02821); [Paul & Chen, 2022](https://arxiv.org/abs/2105.07581)) — understanding *why* is an open research question this thesis addresses.

**3. Test-Time Adaptation**
TENT ([Wang et al., 2021](https://arxiv.org/abs/2006.10726)) and subsequent TTA methods demonstrate that adapting normalization layers at inference significantly closes the clean→corrupted accuracy gap without any labeled test data.

---

## ✦ Dependencies

```toml
[dependencies]
torch       = ">=2.0"
torchvision = ">=0.15"
timm        = ">=0.9"
einops      = ">=0.7"
numpy       = ">=1.24"
tqdm        = ">=4.65"
```

---

## ✦ About

<div align="center">

**Asfand Yar** · BSc Computer Science · University of Debrecen, Hungary

*Thesis project 2025–2026 · Specialization: Generative & Agentic AI Systems*

[![GitHub](https://img.shields.io/badge/GitHub-asfandyar--prog-181717?style=flat-square&logo=github)](https://github.com/asfandyar-prog)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Asfand%20Yar-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/asfand-yar-3966b8291)
[![Email](https://img.shields.io/badge/Email-yarasfand886%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:yarasfand886@gmail.com)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

*If this work is useful to your research, a ⭐ helps the project grow.*

</div>
