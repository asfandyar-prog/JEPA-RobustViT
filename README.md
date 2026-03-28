<div align="center">

<!-- Animated Header Banner -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=JEPA-RobustViT&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Predictive%20Self-Supervised%20Vision%20Transformers%20under%20Distribution%20Shift&descAlignY=58&descSize=16&descColor=a78bfa"/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=JEPA-RobustViT&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Predictive%20Self-Supervised%20Vision%20Transformers%20under%20Distribution%20Shift&descAlignY=58&descSize=16&descColor=a78bfa" alt="Header"/>
</picture>

<br/>

<!-- Animated Typing SVG -->
<a href="https://github.com/asfandyar-prog/JEPA-RobustViT">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=18&duration=3000&pause=1000&color=A78BFA&center=true&vCenter=true&multiline=true&width=700&height=80&lines=BSc+Thesis+%7C+Asfand+Yar+%7C+University+of+Debrecen+2026;JEPA+%2B+ViT+%2B+Test-Time+Adaptation+%3D+Robust+Vision" alt="Typing SVG"/>
</a>

<br/><br/>

<!-- Badges Row 1 -->
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/timm-ViT-7C3AED?style=for-the-badge&logo=huggingface&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/>

<!-- Badges Row 2 -->
<img src="https://img.shields.io/badge/Status-Active%20Research-f59e0b?style=for-the-badge&logo=git&logoColor=white"/>
<img src="https://img.shields.io/badge/ImageNet--C-Robustness-ec4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Self--Supervised-JEPA-06b6d4?style=for-the-badge"/>

</div>

---

<!-- Architecture Diagram — Animated SVG -->
<div align="center">

## ⚡ Architecture at a Glance

```
Raw Image  ─────────────────────────────────────────────────────────────────►  Prediction
    │                                                                               ▲
    ▼                                                                               │
┌──────────┐    ┌─────────────────────────────────────────────────┐    ┌──────────────────┐
│  Patch   │    │           Vision Transformer Backbone            │    │   JEPA Predictive│
│  Embed   │───►│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │───►│       Head       │
│  + Pos   │    │  │ MHA  │►│ MHA  │►│ MHA  │►│ MHA  │►│ MHA  │ │    │   (Target Enc.)  │
└──────────┘    │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │    └──────────────────┘
                │     L1       L2       L3       L4   ...  L12    │             │
                └─────────────────────────────────────────────────┘             ▼
                                        │                               ┌──────────────┐
                                        ▼                               │  Test-Time   │
                            ┌────────────────────┐                      │  Adaptation  │
                            │  CLS Token / Feat  │                      │ (Entropy Min)│
                            └────────────────────┘                      └──────────────┘
```

</div>

---

<div align="center">

## 🧠 Animated Transformer Core

</div>

<!-- Pure SVG animated transformer diagram -->
<div align="center">

<svg width="860" height="380" viewBox="0 0 860 380" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;">
  <defs>
    <!-- Background gradient -->
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0f0c29"/>
      <stop offset="50%" style="stop-color:#302b63"/>
      <stop offset="100%" style="stop-color:#24243e"/>
    </linearGradient>

    <!-- Purple glow gradient for blocks -->
    <linearGradient id="blockGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#7c3aed;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#4c1d95;stop-opacity:0.9"/>
    </linearGradient>

    <!-- Cyan glow for JEPA -->
    <linearGradient id="jepaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#06b6d4;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#0e7490;stop-opacity:0.9"/>
    </linearGradient>

    <!-- Pink glow for TTA -->
    <linearGradient id="ttaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ec4899;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#9d174d;stop-opacity:0.9"/>
    </linearGradient>

    <!-- Amber for patch embed -->
    <linearGradient id="patchGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#b45309;stop-opacity:0.9"/>
    </linearGradient>

    <!-- Glow filters -->
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="strongGlow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>

    <!-- Animated pulse -->
    <style>
      .pulse { animation: pulse 2s ease-in-out infinite alternate; }
      .pulse2 { animation: pulse 2s ease-in-out infinite alternate; animation-delay: 0.4s; }
      .pulse3 { animation: pulse 2s ease-in-out infinite alternate; animation-delay: 0.8s; }
      .pulse4 { animation: pulse 2s ease-in-out infinite alternate; animation-delay: 1.2s; }
      .pulse5 { animation: pulse 2s ease-in-out infinite alternate; animation-delay: 1.6s; }
      @keyframes pulse { from { opacity: 0.6; } to { opacity: 1.0; } }

      .flowDot { animation: flowAnim 2.5s linear infinite; }
      .flowDot2 { animation: flowAnim 2.5s linear infinite; animation-delay: 0.8s; }
      .flowDot3 { animation: flowAnim 2.5s linear infinite; animation-delay: 1.6s; }
      @keyframes flowAnim {
        0% { opacity: 0; transform: translateX(0px); }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { opacity: 0; transform: translateX(480px); }
      }

      .scanLine { animation: scan 3s ease-in-out infinite; }
      @keyframes scan {
        0%, 100% { transform: translateY(0px); opacity: 0.3; }
        50% { transform: translateY(260px); opacity: 0.8; }
      }

      .titleFade { animation: fadeIn 1.5s ease-in forwards; }
      @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

      .attnLine { stroke-dasharray: 4 3; animation: dash 1.5s linear infinite; }
      @keyframes dash { to { stroke-dashoffset: -14; } }

      .sparkle { animation: sparkleAnim 2s ease-in-out infinite; }
      @keyframes sparkleAnim {
        0%, 100% { opacity: 0.2; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
      }
    </style>
  </defs>

  <!-- Background -->
  <rect width="860" height="380" fill="url(#bg)" rx="16"/>

  <!-- Subtle scan line effect -->
  <rect class="scanLine" x="0" y="0" width="860" height="2" fill="#a78bfa" opacity="0.3"/>

  <!-- Title -->
  <text class="titleFade" x="430" y="30" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="13" font-weight="bold" letter-spacing="3">JEPA-RobustViT  ·  ARCHITECTURE OVERVIEW</text>
  <line x1="60" y1="40" x2="800" y2="40" stroke="#7c3aed" stroke-width="0.5" opacity="0.4"/>

  <!-- === PATCH EMBED BLOCK === -->
  <rect class="pulse" x="30" y="80" width="90" height="200" rx="10" fill="url(#patchGrad)" filter="url(#glow)"/>
  <text x="75" y="168" text-anchor="middle" fill="white" font-family="monospace" font-size="10" font-weight="bold">PATCH</text>
  <text x="75" y="182" text-anchor="middle" fill="white" font-family="monospace" font-size="10" font-weight="bold">EMBED</text>
  <text x="75" y="200" text-anchor="middle" fill="#fde68a" font-family="monospace" font-size="8">+ POS ENC</text>
  <!-- Grid icon inside -->
  <rect x="50" y="90" width="8" height="8" rx="1" fill="#fde68a" opacity="0.6"/>
  <rect x="62" y="90" width="8" height="8" rx="1" fill="#fde68a" opacity="0.6"/>
  <rect x="74" y="90" width="8" height="8" rx="1" fill="#fde68a" opacity="0.6"/>
  <rect x="86" y="90" width="8" height="8" rx="1" fill="#fde68a" opacity="0.6"/>
  <rect x="50" y="102" width="8" height="8" rx="1" fill="#fde68a" opacity="0.4"/>
  <rect x="62" y="102" width="8" height="8" rx="1" fill="#fde68a" opacity="0.6"/>
  <rect x="74" y="102" width="8" height="8" rx="1" fill="#fde68a" opacity="0.4"/>
  <rect x="86" y="102" width="8" height="8" rx="1" fill="#fde68a" opacity="0.8"/>
  <rect x="50" y="114" width="8" height="8" rx="1" fill="#fde68a" opacity="0.8"/>
  <rect x="62" y="114" width="8" height="8" rx="1" fill="#fde68a" opacity="0.3"/>
  <rect x="74" y="114" width="8" height="8" rx="1" fill="#fde68a" opacity="0.7"/>
  <rect x="86" y="114" width="8" height="8" rx="1" fill="#fde68a" opacity="0.5"/>
  <!-- Label below -->
  <text x="75" y="300" text-anchor="middle" fill="#f59e0b" font-family="monospace" font-size="9">Input</text>
  <text x="75" y="312" text-anchor="middle" fill="#f59e0b" font-family="monospace" font-size="9">Tokens</text>

  <!-- Flow arrow -->
  <line x1="122" y1="180" x2="148" y2="180" stroke="#a78bfa" stroke-width="1.5"/>
  <polygon points="148,175 158,180 148,185" fill="#a78bfa"/>
  <!-- Animated flow dot -->
  <circle class="flowDot" cx="125" cy="180" r="3" fill="#f59e0b" opacity="0"/>

  <!-- === TRANSFORMER BLOCKS (L1-L4 visible, rest implied) === -->
  <!-- Block L1 -->
  <rect class="pulse" x="160" y="60" width="68" height="240" rx="8" fill="url(#blockGrad)" filter="url(#glow)"/>
  <text x="194" y="148" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">MULTI</text>
  <text x="194" y="160" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">HEAD</text>
  <text x="194" y="172" text-anchor="middle" fill="#c4b5fd" font-family="monospace" font-size="9" font-weight="bold">ATTN</text>
  <!-- Attention lines inside -->
  <line class="attnLine" x1="170" y1="185" x2="220" y2="185" stroke="#c4b5fd" stroke-width="1" opacity="0.5"/>
  <line class="attnLine" x1="170" y1="195" x2="220" y2="195" stroke="#c4b5fd" stroke-width="1" opacity="0.4"/>
  <line class="attnLine" x1="170" y1="205" x2="220" y2="205" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <text x="194" y="230" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ FFN</text>
  <text x="194" y="242" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ LN</text>
  <text x="194" y="320" text-anchor="middle" fill="#7c3aed" font-family="monospace" font-size="9">L1</text>

  <!-- Connector -->
  <line x1="230" y1="180" x2="244" y2="180" stroke="#a78bfa" stroke-width="1.5"/>
  <polygon points="244,175 254,180 244,185" fill="#a78bfa"/>
  <circle class="flowDot2" cx="232" cy="180" r="3" fill="#a78bfa" opacity="0"/>

  <!-- Block L2 -->
  <rect class="pulse2" x="256" y="60" width="68" height="240" rx="8" fill="url(#blockGrad)" filter="url(#glow)"/>
  <text x="290" y="148" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">MULTI</text>
  <text x="290" y="160" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">HEAD</text>
  <text x="290" y="172" text-anchor="middle" fill="#c4b5fd" font-family="monospace" font-size="9" font-weight="bold">ATTN</text>
  <line class="attnLine" x1="266" y1="185" x2="316" y2="185" stroke="#c4b5fd" stroke-width="1" opacity="0.5"/>
  <line class="attnLine" x1="266" y1="195" x2="316" y2="195" stroke="#c4b5fd" stroke-width="1" opacity="0.4"/>
  <line class="attnLine" x1="266" y1="205" x2="316" y2="205" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <text x="290" y="230" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ FFN</text>
  <text x="290" y="242" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ LN</text>
  <text x="290" y="320" text-anchor="middle" fill="#7c3aed" font-family="monospace" font-size="9">L2</text>

  <line x1="326" y1="180" x2="340" y2="180" stroke="#a78bfa" stroke-width="1.5"/>
  <polygon points="340,175 350,180 340,185" fill="#a78bfa"/>
  <circle class="flowDot3" cx="328" cy="180" r="3" fill="#a78bfa" opacity="0"/>

  <!-- Block L3 -->
  <rect class="pulse3" x="352" y="60" width="68" height="240" rx="8" fill="url(#blockGrad)" filter="url(#glow)"/>
  <text x="386" y="148" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">MULTI</text>
  <text x="386" y="160" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">HEAD</text>
  <text x="386" y="172" text-anchor="middle" fill="#c4b5fd" font-family="monospace" font-size="9" font-weight="bold">ATTN</text>
  <line class="attnLine" x1="362" y1="185" x2="412" y2="185" stroke="#c4b5fd" stroke-width="1" opacity="0.5"/>
  <line class="attnLine" x1="362" y1="195" x2="412" y2="195" stroke="#c4b5fd" stroke-width="1" opacity="0.4"/>
  <line class="attnLine" x1="362" y1="205" x2="412" y2="205" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <text x="386" y="230" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ FFN</text>
  <text x="386" y="242" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ LN</text>
  <text x="386" y="320" text-anchor="middle" fill="#7c3aed" font-family="monospace" font-size="9">L3</text>

  <line x1="422" y1="180" x2="436" y2="180" stroke="#a78bfa" stroke-width="1.5"/>
  <polygon points="436,175 446,180 436,185" fill="#a78bfa"/>

  <!-- Ellipsis blocks -->
  <circle class="pulse4" cx="464" cy="174" r="4" fill="#6d28d9"/>
  <circle class="pulse5" cx="476" cy="174" r="4" fill="#6d28d9"/>
  <circle class="pulse" cx="488" cy="174" r="4" fill="#6d28d9"/>
  <text x="476" y="200" text-anchor="middle" fill="#7c3aed" font-family="monospace" font-size="8">L4..L12</text>

  <line x1="498" y1="180" x2="512" y2="180" stroke="#a78bfa" stroke-width="1.5"/>
  <polygon points="512,175 522,180 512,185" fill="#a78bfa"/>

  <!-- Block L12 (final) -->
  <rect class="pulse4" x="524" y="60" width="68" height="240" rx="8" fill="url(#blockGrad)" filter="url(#glow)"/>
  <text x="558" y="148" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">MULTI</text>
  <text x="558" y="160" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">HEAD</text>
  <text x="558" y="172" text-anchor="middle" fill="#c4b5fd" font-family="monospace" font-size="9" font-weight="bold">ATTN</text>
  <line class="attnLine" x1="534" y1="185" x2="584" y2="185" stroke="#c4b5fd" stroke-width="1" opacity="0.5"/>
  <line class="attnLine" x1="534" y1="195" x2="584" y2="195" stroke="#c4b5fd" stroke-width="1" opacity="0.4"/>
  <line class="attnLine" x1="534" y1="205" x2="584" y2="205" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <text x="558" y="230" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ FFN</text>
  <text x="558" y="242" text-anchor="middle" fill="#a78bfa" font-family="monospace" font-size="8">+ LN</text>
  <text x="558" y="320" text-anchor="middle" fill="#7c3aed" font-family="monospace" font-size="9">L12</text>

  <!-- CLS Token branch (downward) -->
  <line x1="558" y1="302" x2="558" y2="340" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4,2"/>
  <text x="558" y="358" text-anchor="middle" fill="#c4b5fd" font-family="monospace" font-size="8">[CLS]</text>

  <!-- Arrow to JEPA -->
  <line x1="594" y1="130" x2="628" y2="130" stroke="#06b6d4" stroke-width="1.5"/>
  <polygon points="628,125 638,130 628,135" fill="#06b6d4"/>

  <!-- === JEPA HEAD === -->
  <rect class="pulse2" x="640" y="80" width="90" height="120" rx="10" fill="url(#jepaGrad)" filter="url(#glow)"/>
  <text x="685" y="112" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">JEPA</text>
  <text x="685" y="124" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">PRED</text>
  <text x="685" y="136" text-anchor="middle" fill="#a5f3fc" font-family="monospace" font-size="9" font-weight="bold">HEAD</text>
  <text x="685" y="156" text-anchor="middle" fill="#e0f2fe" font-family="monospace" font-size="8">Target Enc.</text>
  <text x="685" y="168" text-anchor="middle" fill="#e0f2fe" font-family="monospace" font-size="8">Context Enc.</text>
  <text x="685" y="180" text-anchor="middle" fill="#e0f2fe" font-family="monospace" font-size="8">Patch Predict</text>
  <text x="685" y="220" text-anchor="middle" fill="#06b6d4" font-family="monospace" font-size="9">SSL Obj.</text>

  <!-- Arrow to TTA -->
  <line x1="685" y1="202" x2="685" y2="228" stroke="#ec4899" stroke-width="1.5"/>
  <polygon points="680,228 685,238 690,228" fill="#ec4899"/>

  <!-- === TTA BLOCK === -->
  <rect class="pulse3" x="640" y="240" width="90" height="100" rx="10" fill="url(#ttaGrad)" filter="url(#glow)"/>
  <text x="685" y="270" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">TEST-TIME</text>
  <text x="685" y="282" text-anchor="middle" fill="white" font-family="monospace" font-size="9" font-weight="bold">ADAPT</text>
  <text x="685" y="298" text-anchor="middle" fill="#fce7f3" font-family="monospace" font-size="8">Entropy Min</text>
  <text x="685" y="310" text-anchor="middle" fill="#fce7f3" font-family="monospace" font-size="8">BN Adapt</text>
  <text x="685" y="322" text-anchor="middle" fill="#fce7f3" font-family="monospace" font-size="8">Aug. Avg.</text>

  <!-- Sparkles for effect -->
  <circle class="sparkle" cx="140" cy="70" r="3" fill="#f59e0b"/>
  <circle class="sparkle" cx="618" cy="70" r="3" fill="#06b6d4"/>
  <circle class="sparkle" cx="750" cy="180" r="3" fill="#ec4899"/>
  <circle class="sparkle" cx="50" cy="290" r="2" fill="#a78bfa"/>

  <!-- Legend -->
  <rect x="30" y="350" width="10" height="10" rx="2" fill="url(#patchGrad)"/>
  <text x="45" y="360" fill="#f59e0b" font-family="monospace" font-size="8">Patch Embedding</text>
  <rect x="160" y="350" width="10" height="10" rx="2" fill="url(#blockGrad)"/>
  <text x="175" y="360" fill="#a78bfa" font-family="monospace" font-size="8">Transformer Layer</text>
  <rect x="330" y="350" width="10" height="10" rx="2" fill="url(#jepaGrad)"/>
  <text x="345" y="360" fill="#06b6d4" font-family="monospace" font-size="8">JEPA Head</text>
  <rect x="440" y="350" width="10" height="10" rx="2" fill="url(#ttaGrad)"/>
  <text x="455" y="360" fill="#ec4899" font-family="monospace" font-size="8">Test-Time Adaptation</text>
</svg>

</div>

---

## 📌 Abstract

> **JEPA-RobustViT** investigates whether *predictive self-supervised pre-training* — specifically a Joint Embedding Predictive Architecture over Vision Transformer representations — confers measurably improved **distributional robustness** under real-world corruption and domain shift. The system couples a frozen or fine-tuned ViT backbone with entropy-minimisation-based test-time adaptation, evaluated against ImageNet-C and ImageNet-R benchmarks. This constitutes the **BSc thesis** of Asfand Yar at the University of Debrecen (2025–2026).

---

## 🗺️ Research Roadmap

```
Phase 1 ─ Backbone         Phase 2 ─ JEPA SSL          Phase 3 ─ TTA               Phase 4 ─ Eval
───────────────────         ──────────────────           ────────────────            ──────────────
 ✅ ViT-S/16 (timm)          ✅ Context encoder           ✅ Entropy min.             🔄 ImageNet-C
 ✅ CIFAR-10 baseline         ✅ Target encoder (EMA)      ✅ BN adaptation            🔄 ImageNet-R
 ✅ Patch embedding           ✅ Predictive head           🔄 Aug. consistency         ⬜ OOD ablation
 ✅ CLS token probing         🔄 JEPA loss (latent)       ⬜ LAME / CoTTA             ⬜ mCE metric
```

---

## 🔬 Key Components

### 🧱 Backbone — `src/backbone.py`

A **`timm`-loaded ViT** (`vit_small_patch16_224`) frozen after DINO pre-training. Features are extracted from the `[CLS]` token for downstream evaluation and JEPA target prediction.

```python
import timm

backbone = timm.create_model(
    "vit_small_patch16_224",
    pretrained=True,
    num_classes=0       # headless — features only
)
```

### 🎯 JEPA Predictive Head — `src/jepa_head.py`

Implements the **context → target patch prediction** objective in representation space:
- **Context Encoder** — processes visible (unmasked) patches
- **Target Encoder** — EMA copy of context encoder; produces prediction targets
- **Predictor** — narrow Transformer maps context features to target features

```python
# Core JEPA objective
with torch.no_grad():
    target_features = target_encoder(target_patches)    # EMA targets

pred_features = predictor(context_features, mask_ids)   # masked prediction
loss = F.smooth_l1_loss(pred_features, target_features) # latent-space loss
```

### 🔄 Test-Time Adaptation — `src/tta.py`

At inference, the model adapts *on-the-fly* to corrupted inputs without any labelled data:

| Strategy | Mechanism | Benefit |
|---|---|---|
| **Entropy Minimisation** | Minimise `H(p)` on augmented views | Pulls uncertain predictions to confident regions |
| **BN Adaptation** | Re-estimate μ/σ on test batch | Corrects covariate shift in batch statistics |
| **Augmentation Averaging** | Ensemble predictions over augmentations | Smooths corruption artefacts |

```python
# Entropy minimisation (TENT-style)
def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * probs.log()).sum(dim=-1).mean()
```

### 🩺 Medical Dataset Loaders — `src/datasets.py`

Extended dataset support for domain-shift experiments:

```python
supported_datasets = [
    "CIFAR-10",       # baseline
    "CIFAR-10-C",     # corruption robustness
    "ImageNet-C",     # 15 corruption types × 5 severities
    "ImageNet-R",     # rendition domain shift
    "MedMNIST-*",     # medical imaging OOD
]
```

---

## 📊 Baseline Results

| Model | CIFAR-10 (Clean) | Notes |
|---|---|---|
| ViT-S/16 (linear probe) | ~85.2% | timm DINO pretrain |
| ViT-S/16 + TTA (entropy) | ~86.1% | entropy min, 10-aug |
| JEPA-ViT (ours, in progress) | 🔄 | latent pred. SSL |

> Full ImageNet-C/R results will populate in `results/baselines.md` as evaluation runs complete.

---

## 🚀 Quickstart

**Requirements:** Python 3.11+, CUDA 11.8+ (or CPU for small-scale experiments)

```bash
# 1. Clone the repository
git clone https://github.com/asfandyar-prog/JEPA-RobustViT.git
cd JEPA-RobustViT

# 2. Install dependencies (uv recommended)
pip install uv
uv sync

# — or with pip directly —
pip install -r requirements.txt

# 3. Run CIFAR-10 baseline
python main.py --mode baseline --dataset cifar10

# 4. Run with TTA enabled
python main.py --mode tta --dataset cifar10-c --severity 3

# 5. Run JEPA pre-training
python main.py --mode jepa_pretrain --dataset imagenet --epochs 100
```

---

## 🗂️ Repository Structure

```
JEPA-RobustViT/
│
├── src/
│   ├── backbone.py          # ViT backbone (timm) + feature extraction
│   ├── classifier.py        # Linear probe / fine-tune head
│   ├── jepa_head.py         # Context/target encoders + predictive head
│   ├── tta.py               # Test-time adaptation strategies
│   └── datasets.py          # Loaders: CIFAR, ImageNet-C/R, MedMNIST
│
├── scripts/
│   ├── test_backbone.py     # Backbone sanity checks
│   └── eval_tta.py          # TTA evaluation pipeline
│
├── results/
│   └── baselines.md         # Tracked experiment results
│
├── main.py                  # Entry point — train / eval / pretrain
├── pyproject.toml           # Project metadata (uv)
├── requirements.txt         # pip-compatible dependencies
└── README.md
```

---

## 🧪 Evaluation Protocol

Distribution shift robustness is measured via **mean Corruption Error (mCE)**:

```
mCE = (1/15) Σ_c  [E_model(c) / E_AlexNet(c)]

where c ∈ {Gaussian noise, shot noise, impulse noise, defocus blur,
           glass blur, motion blur, zoom blur, snow, frost, fog,
           brightness, contrast, elastic, pixelate, JPEG}
```

Severity levels 1–5 are averaged per corruption type.

---

## 🔭 Research Questions

| # | Question | Status |
|---|---|---|
| RQ1 | Does JEPA pre-training outperform vanilla supervised ViT on ImageNet-C mCE? | 🔄 In progress |
| RQ2 | Does TTA further close the gap independent of SSL objective? | 🔄 In progress |
| RQ3 | Is predictive SSL complementary to entropy-min TTA or redundant? | ⬜ Planned |
| RQ4 | Do gains transfer to medical imaging domain shift? | ⬜ Planned |

---

## 📚 References & Related Work

| Paper | Relevance |
|---|---|
| [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243) | JEPA objective in image space |
| [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294) | ViT self-supervised pre-training |
| [TENT (Wang et al., 2021)](https://arxiv.org/abs/2006.10726) | Entropy-min TTA baseline |
| [ImageNet-C (Hendrycks & Dietterich, 2019)](https://arxiv.org/abs/1903.12261) | Corruption robustness benchmark |
| [ImageNet-R (Hendrycks et al., 2021)](https://arxiv.org/abs/2006.16241) | Rendition domain shift |

---

## 👨‍💻 Author

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:302b63,100:24243e&height=2&width=600" width="600"/>

**Asfand Yar**
BSc Computer Science · University of Debrecen · 2025–2026

[![GitHub](https://img.shields.io/badge/GitHub-asfandyar--prog-181717?style=flat-square&logo=github)](https://github.com/asfandyar-prog)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-asfand--yar-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/asfand-yar-3966b8291)
[![Email](https://img.shields.io/badge/Email-yarasfand886@gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:yarasfand886@gmail.com)

*Co-Lead, Google Developer Groups Debrecen · VP, International Students' Union*

</div>

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer"/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" alt="Footer"/>
</picture>

*"Robustness is not a feature. It is a prerequisite."*

![Visitor Count](https://komarev.com/ghpvc/?username=asfandyar-prog&color=7c3aed&style=flat-square&label=Profile+Views)
