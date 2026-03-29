<svg width="900" height="1020" viewBox="0 0 900 1020" xmlns="http://www.w3.org/2000/svg" font-family="'Courier New', Courier, monospace">
  <defs>
    <!-- ── Filters ── -->
    <filter id="glow-b" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-sm" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-cyan" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.5"/>
    </filter>

    <!-- ── Gradients ── -->
    <linearGradient id="bg" x1="0" y1="0" x2="900" y2="1020" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#05050f"/>
      <stop offset="50%" stop-color="#080818"/>
      <stop offset="100%" stop-color="#060612"/>
    </linearGradient>
    <linearGradient id="embed-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1a4a7a"/><stop offset="100%" stop-color="#0a2040"/>
    </linearGradient>
    <linearGradient id="ln-g" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#4f46e5"/>
    </linearGradient>
    <linearGradient id="attn-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1e1060"/><stop offset="100%" stop-color="#0d0830"/>
    </linearGradient>
    <linearGradient id="q-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2563eb"/><stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="k-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#059669"/><stop offset="100%" stop-color="#047857"/>
    </linearGradient>
    <linearGradient id="v-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#dc2626"/><stop offset="100%" stop-color="#b91c1c"/>
    </linearGradient>
    <linearGradient id="ffn-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0f3a2a"/><stop offset="100%" stop-color="#071f15"/>
    </linearGradient>
    <linearGradient id="out-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1a0a40"/><stop offset="100%" stop-color="#0a0520"/>
    </linearGradient>
    <linearGradient id="softmax-g" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#06b6d4"/>
    </linearGradient>
    <linearGradient id="jepa-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#064e3b"/><stop offset="100%" stop-color="#022c22"/>
    </linearGradient>
    <linearGradient id="tta-g" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#431407"/><stop offset="100%" stop-color="#1a0803"/>
    </linearGradient>
    <linearGradient id="flow-v" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#06b6d4" stop-opacity="0"/>
      <stop offset="50%" stop-color="#06b6d4" stop-opacity="1"/>
      <stop offset="100%" stop-color="#06b6d4" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="flow-purple" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#a855f7" stop-opacity="0"/>
      <stop offset="50%" stop-color="#a855f7" stop-opacity="1"/>
      <stop offset="100%" stop-color="#a855f7" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="flow-green" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#10b981" stop-opacity="0"/>
      <stop offset="50%" stop-color="#10b981" stop-opacity="1"/>
      <stop offset="100%" stop-color="#10b981" stop-opacity="0"/>
    </linearGradient>

    <!-- ── Pattern: grid dots ── -->
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <circle cx="0.5" cy="0.5" r="0.8" fill="#1e1e4a" opacity="0.7"/>
    </pattern>

    <!-- ── Marker: arrowhead ── -->
    <marker id="arr-cyan" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#06b6d4"/>
    </marker>
    <marker id="arr-purple" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#a855f7"/>
    </marker>
    <marker id="arr-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#10b981"/>
    </marker>
    <marker id="arr-orange" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#f97316"/>
    </marker>
    <marker id="arr-white" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>
    </marker>
  </defs>

  <!-- ══════════════════════════════════════════
       BACKGROUND
  ══════════════════════════════════════════ -->
  <rect width="900" height="1020" fill="url(#bg)" rx="16"/>
  <rect width="900" height="1020" fill="url(#grid)" rx="16" opacity="1"/>

  <!-- Ambient glow blobs -->
  <ellipse cx="200" cy="300" rx="180" ry="120" fill="#3730a3" opacity="0.05"/>
  <ellipse cx="700" cy="600" rx="160" ry="100" fill="#0e7490" opacity="0.06"/>
  <ellipse cx="450" cy="800" rx="200" ry="100" fill="#065f46" opacity="0.05"/>

  <!-- ══════════════════════════════════════════
       TITLE
  ══════════════════════════════════════════ -->
  <text x="450" y="36" text-anchor="middle" font-size="13" font-weight="bold" fill="#e2e8f0" letter-spacing="4">JEPA-RobustViT  ·  Transformer Architecture</text>
  <line x1="120" y1="44" x2="780" y2="44" stroke="#1e293b" stroke-width="1"/>

  <!-- ══════════════════════════════════════════
       SECTION 1 — INPUT TOKENS  (y: 58–138)
  ══════════════════════════════════════════ -->
  <!-- Token sequence: 7 patch tokens + 1 CLS -->
  <text x="28" y="70" font-size="9" fill="#64748b" letter-spacing="1">INPUT</text>

  <!-- CLS token -->
  <rect x="28" y="76" width="52" height="52" rx="6" fill="#312e81" stroke="#6366f1" stroke-width="1.5" filter="url(#glow-sm)"/>
  <text x="54" y="97" text-anchor="middle" font-size="8" fill="#a5b4fc" font-weight="bold">[CLS]</text>
  <text x="54" y="109" text-anchor="middle" font-size="7" fill="#818cf8">token</text>
  <text x="54" y="120" text-anchor="middle" font-size="7" fill="#4338ca">pos=0</text>

  <!-- Patch tokens -->
  <!-- Colors cycle through the rainbow to indicate different patches -->
  <g>
    <!-- patch 1 -->
    <rect x="90" y="76" width="52" height="52" rx="6" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.2"/>
    <rect x="90" y="76" width="52" height="18" rx="3" fill="#1d4ed8" opacity="0.5"/>
    <text x="116" y="97" text-anchor="middle" font-size="8" fill="#93c5fd" font-weight="bold">P₁</text>
    <text x="116" y="109" text-anchor="middle" font-size="7" fill="#60a5fa">patch</text>
    <text x="116" y="120" text-anchor="middle" font-size="7" fill="#2563eb">pos=1</text>

    <!-- patch 2 -->
    <rect x="152" y="76" width="52" height="52" rx="6" fill="#1e3a3a" stroke="#06b6d4" stroke-width="1.2"/>
    <rect x="152" y="76" width="52" height="18" rx="3" fill="#0e7490" opacity="0.5"/>
    <text x="178" y="97" text-anchor="middle" font-size="8" fill="#67e8f9" font-weight="bold">P₂</text>
    <text x="178" y="109" text-anchor="middle" font-size="7" fill="#22d3ee">patch</text>
    <text x="178" y="120" text-anchor="middle" font-size="7" fill="#0891b2">pos=2</text>

    <!-- patch 3 -->
    <rect x="214" y="76" width="52" height="52" rx="6" fill="#1e3a20" stroke="#22c55e" stroke-width="1.2"/>
    <rect x="214" y="76" width="52" height="18" rx="3" fill="#166534" opacity="0.5"/>
    <text x="240" y="97" text-anchor="middle" font-size="8" fill="#86efac" font-weight="bold">P₃</text>
    <text x="240" y="109" text-anchor="middle" font-size="7" fill="#4ade80">patch</text>
    <text x="240" y="120" text-anchor="middle" font-size="7" fill="#16a34a">pos=3</text>

    <!-- patch 4 -->
    <rect x="276" y="76" width="52" height="52" rx="6" fill="#3a3a1e" stroke="#eab308" stroke-width="1.2"/>
    <rect x="276" y="76" width="52" height="18" rx="3" fill="#713f12" opacity="0.5"/>
    <text x="302" y="97" text-anchor="middle" font-size="8" fill="#fde68a" font-weight="bold">P₄</text>
    <text x="302" y="109" text-anchor="middle" font-size="7" fill="#fbbf24">patch</text>
    <text x="302" y="120" text-anchor="middle" font-size="7" fill="#d97706">pos=4</text>

    <!-- patch 5 -->
    <rect x="338" y="76" width="52" height="52" rx="6" fill="#3a1e1e" stroke="#ef4444" stroke-width="1.2"/>
    <rect x="338" y="76" width="52" height="18" rx="3" fill="#7f1d1d" opacity="0.5"/>
    <text x="364" y="97" text-anchor="middle" font-size="8" fill="#fca5a5" font-weight="bold">P₅</text>
    <text x="364" y="109" text-anchor="middle" font-size="7" fill="#f87171">patch</text>
    <text x="364" y="120" text-anchor="middle" font-size="7" fill="#dc2626">pos=5</text>

    <!-- patch 6 -->
    <rect x="400" y="76" width="52" height="52" rx="6" fill="#2e1e3a" stroke="#a855f7" stroke-width="1.2"/>
    <rect x="400" y="76" width="52" height="18" rx="3" fill="#581c87" opacity="0.5"/>
    <text x="426" y="97" text-anchor="middle" font-size="8" fill="#d8b4fe" font-weight="bold">P₆</text>
    <text x="426" y="109" text-anchor="middle" font-size="7" fill="#c084fc">patch</text>
    <text x="426" y="120" text-anchor="middle" font-size="7" fill="#9333ea">pos=6</text>

    <!-- dots -->
    <circle cx="476" cy="102" r="3" fill="#334155"/>
    <circle cx="490" cy="102" r="3" fill="#334155"/>
    <circle cx="504" cy="102" r="3" fill="#334155"/>
  </g>

  <!-- Positional encoding label -->
  <text x="28" y="140" font-size="9" fill="#64748b" letter-spacing="1">+ POSITIONAL ENCODING  ⊕  LINEAR PROJECTION  →  d_model = 384</text>

  <!-- Flow arrow down -->
  <line x1="450" y1="148" x2="450" y2="168" stroke="#1e293b" stroke-width="1.5" marker-end="url(#arr-white)"/>
  <rect x="430" y="150" width="40" height="18" fill="url(#flow-v)" rx="4" opacity="0.7">
    <animate attributeName="y" values="148;168;148" dur="3s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0;0.8;0" dur="3s" repeatCount="indefinite"/>
  </rect>

  <!-- ══════════════════════════════════════════
       TRANSFORMER BLOCK OUTER FRAME (y: 172–750)
  ══════════════════════════════════════════ -->
  <rect x="18" y="172" width="864" height="582" rx="14" fill="none" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="6,4"/>
  <text x="30" y="188" font-size="9" fill="#475569" letter-spacing="2">TRANSFORMER ENCODER BLOCK  ×L</text>

  <!-- ── RESIDUAL STREAM (vertical backbone, left side) ── -->
  <!-- The main vertical stream -->
  <rect x="56" y="196" width="16" height="548" rx="8" fill="#0f172a" stroke="#1e3a5f" stroke-width="1"/>
  <!-- animated pulse along stream -->
  <rect x="58" y="200" width="12" height="60" rx="6" fill="url(#flow-v)" opacity="0.6">
    <animate attributeName="y" values="196;680;196" dur="4s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0;0.7;0.7;0" dur="4s" repeatCount="indefinite"/>
  </rect>
  <text x="64" y="477" text-anchor="middle" font-size="8" fill="#334155" transform="rotate(-90,64,477)" letter-spacing="2">RESIDUAL STREAM</text>

  <!-- ══════════════════════════════════════════
       LAYER NORM 1  (y: 196–230)
  ══════════════════════════════════════════ -->
  <rect x="82" y="196" width="800" height="42" rx="8" fill="#0f0a2a" stroke="#7c3aed" stroke-width="1.2"/>
  <!-- LN bar as a heatmap row -->
  <g>
    <rect x="90" y="204" width="15" height="26" rx="2" fill="#4c1d95" opacity="0.9"/>
    <rect x="108" y="204" width="15" height="26" rx="2" fill="#5b21b6" opacity="0.85"/>
    <rect x="126" y="204" width="15" height="26" rx="2" fill="#6d28d9" opacity="0.9"/>
    <rect x="144" y="204" width="15" height="26" rx="2" fill="#7c3aed" opacity="0.95"/>
    <rect x="162" y="204" width="15" height="26" rx="2" fill="#8b5cf6" opacity="1"/>
    <rect x="180" y="204" width="15" height="26" rx="2" fill="#7c3aed" opacity="0.95"/>
    <rect x="198" y="204" width="15" height="26" rx="2" fill="#6d28d9" opacity="0.9"/>
    <rect x="216" y="204" width="15" height="26" rx="2" fill="#5b21b6" opacity="0.8"/>
    <rect x="234" y="204" width="15" height="26" rx="2" fill="#4c1d95" opacity="0.75"/>
  </g>
  <text x="440" y="222" text-anchor="middle" font-size="11" fill="#c4b5fd" font-weight="bold" letter-spacing="2">LAYER NORM 1</text>
  <text x="750" y="222" text-anchor="middle" font-size="9" fill="#6d28d9">x̂ = (x − μ) / σ · γ + β</text>

  <!-- flow down -->
  <line x1="450" y1="238" x2="450" y2="256" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arr-purple)"/>

  <!-- ══════════════════════════════════════════
       MULTI-HEAD SELF-ATTENTION BLOCK  (y: 260–490)
  ══════════════════════════════════════════ -->
  <rect x="82" y="260" width="800" height="232" rx="10" fill="url(#attn-g)" stroke="#4f46e5" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="440" y="278" text-anchor="middle" font-size="12" fill="#818cf8" font-weight="bold" letter-spacing="3">MULTI-HEAD SELF-ATTENTION   (h=6 heads)</text>

  <!-- ── Q K V PROJECTIONS ── -->
  <!-- Q -->
  <rect x="100" y="286" width="168" height="72" rx="8" fill="url(#q-g)" stroke="#3b82f6" stroke-width="1.2"/>
  <text x="184" y="308" text-anchor="middle" font-size="11" fill="#bfdbfe" font-weight="bold">Query  (Q)</text>
  <!-- Q weight matrix visualization -->
  <g opacity="0.6">
    <rect x="108" y="316" width="10" height="10" rx="1" fill="#93c5fd"/>
    <rect x="122" y="316" width="10" height="10" rx="1" fill="#60a5fa"/>
    <rect x="136" y="316" width="10" height="10" rx="1" fill="#3b82f6"/>
    <rect x="150" y="316" width="10" height="10" rx="1" fill="#2563eb"/>
    <rect x="164" y="316" width="10" height="10" rx="1" fill="#1d4ed8"/>
    <rect x="178" y="316" width="10" height="10" rx="1" fill="#1e40af"/>
    <rect x="192" y="316" width="10" height="10" rx="1" fill="#1d4ed8"/>
    <rect x="206" y="316" width="10" height="10" rx="1" fill="#2563eb"/>
    <rect x="220" y="316" width="10" height="10" rx="1" fill="#3b82f6"/>
    <rect x="108" y="330" width="10" height="10" rx="1" fill="#1d4ed8"/>
    <rect x="122" y="330" width="10" height="10" rx="1" fill="#2563eb"/>
    <rect x="136" y="330" width="10" height="10" rx="1" fill="#3b82f6"/>
    <rect x="150" y="330" width="10" height="10" rx="1" fill="#60a5fa"/>
    <rect x="164" y="330" width="10" height="10" rx="1" fill="#93c5fd"/>
    <rect x="178" y="330" width="10" height="10" rx="1" fill="#60a5fa"/>
    <rect x="192" y="330" width="10" height="10" rx="1" fill="#3b82f6"/>
    <rect x="206" y="330" width="10" height="10" rx="1" fill="#2563eb"/>
    <rect x="220" y="330" width="10" height="10" rx="1" fill="#1d4ed8"/>
    <rect x="108" y="344" width="10" height="10" rx="1" fill="#3b82f6"/>
    <rect x="122" y="344" width="10" height="10" rx="1" fill="#1d4ed8"/>
    <rect x="136" y="344" width="10" height="10" rx="1" fill="#1e40af"/>
    <rect x="150" y="344" width="10" height="10" rx="1" fill="#1d4ed8"/>
    <rect x="164" y="344" width="10" height="10" rx="1" fill="#2563eb"/>
    <rect x="178" y="344" width="10" height="10" rx="1" fill="#3b82f6"/>
    <rect x="192" y="344" width="10" height="10" rx="1" fill="#60a5fa"/>
    <rect x="206" y="344" width="10" height="10" rx="1" fill="#93c5fd"/>
    <rect x="220" y="344" width="10" height="10" rx="1" fill="#bfdbfe"/>
  </g>
  <text x="184" y="368" text-anchor="middle" font-size="8" fill="#60a5fa">W_Q ∈ ℝ^{d×d_k}</text>

  <!-- K -->
  <rect x="284" y="286" width="168" height="72" rx="8" fill="url(#k-g)" stroke="#10b981" stroke-width="1.2"/>
  <text x="368" y="308" text-anchor="middle" font-size="11" fill="#a7f3d0" font-weight="bold">Key  (K)</text>
  <g opacity="0.6">
    <rect x="292" y="316" width="10" height="10" rx="1" fill="#6ee7b7"/>
    <rect x="306" y="316" width="10" height="10" rx="1" fill="#34d399"/>
    <rect x="320" y="316" width="10" height="10" rx="1" fill="#10b981"/>
    <rect x="334" y="316" width="10" height="10" rx="1" fill="#059669"/>
    <rect x="348" y="316" width="10" height="10" rx="1" fill="#047857"/>
    <rect x="362" y="316" width="10" height="10" rx="1" fill="#065f46"/>
    <rect x="376" y="316" width="10" height="10" rx="1" fill="#047857"/>
    <rect x="390" y="316" width="10" height="10" rx="1" fill="#059669"/>
    <rect x="404" y="316" width="10" height="10" rx="1" fill="#10b981"/>
    <rect x="292" y="330" width="10" height="10" rx="1" fill="#059669"/>
    <rect x="306" y="330" width="10" height="10" rx="1" fill="#10b981"/>
    <rect x="320" y="330" width="10" height="10" rx="1" fill="#34d399"/>
    <rect x="334" y="330" width="10" height="10" rx="1" fill="#6ee7b7"/>
    <rect x="348" y="330" width="10" height="10" rx="1" fill="#a7f3d0"/>
    <rect x="362" y="330" width="10" height="10" rx="1" fill="#6ee7b7"/>
    <rect x="376" y="330" width="10" height="10" rx="1" fill="#34d399"/>
    <rect x="390" y="330" width="10" height="10" rx="1" fill="#10b981"/>
    <rect x="404" y="330" width="10" height="10" rx="1" fill="#059669"/>
    <rect x="292" y="344" width="10" height="10" rx="1" fill="#10b981"/>
    <rect x="306" y="344" width="10" height="10" rx="1" fill="#059669"/>
    <rect x="320" y="344" width="10" height="10" rx="1" fill="#047857"/>
    <rect x="334" y="344" width="10" height="10" rx="1" fill="#065f46"/>
    <rect x="348" y="344" width="10" height="10" rx="1" fill="#047857"/>
    <rect x="362" y="344" width="10" height="10" rx="1" fill="#059669"/>
    <rect x="376" y="344" width="10" height="10" rx="1" fill="#10b981"/>
    <rect x="390" y="344" width="10" height="10" rx="1" fill="#34d399"/>
    <rect x="404" y="344" width="10" height="10" rx="1" fill="#6ee7b7"/>
  </g>
  <text x="368" y="368" text-anchor="middle" font-size="8" fill="#34d399">W_K ∈ ℝ^{d×d_k}</text>

  <!-- V -->
  <rect x="468" y="286" width="168" height="72" rx="8" fill="url(#v-g)" stroke="#ef4444" stroke-width="1.2"/>
  <text x="552" y="308" text-anchor="middle" font-size="11" fill="#fecaca" font-weight="bold">Value  (V)</text>
  <g opacity="0.6">
    <rect x="476" y="316" width="10" height="10" rx="1" fill="#fca5a5"/>
    <rect x="490" y="316" width="10" height="10" rx="1" fill="#f87171"/>
    <rect x="504" y="316" width="10" height="10" rx="1" fill="#ef4444"/>
    <rect x="518" y="316" width="10" height="10" rx="1" fill="#dc2626"/>
    <rect x="532" y="316" width="10" height="10" rx="1" fill="#b91c1c"/>
    <rect x="546" y="316" width="10" height="10" rx="1" fill="#991b1b"/>
    <rect x="560" y="316" width="10" height="10" rx="1" fill="#b91c1c"/>
    <rect x="574" y="316" width="10" height="10" rx="1" fill="#dc2626"/>
    <rect x="588" y="316" width="10" height="10" rx="1" fill="#ef4444"/>
    <rect x="476" y="330" width="10" height="10" rx="1" fill="#dc2626"/>
    <rect x="490" y="330" width="10" height="10" rx="1" fill="#ef4444"/>
    <rect x="504" y="330" width="10" height="10" rx="1" fill="#f87171"/>
    <rect x="518" y="330" width="10" height="10" rx="1" fill="#fca5a5"/>
    <rect x="532" y="330" width="10" height="10" rx="1" fill="#fecaca"/>
    <rect x="546" y="330" width="10" height="10" rx="1" fill="#fca5a5"/>
    <rect x="560" y="330" width="10" height="10" rx="1" fill="#f87171"/>
    <rect x="574" y="330" width="10" height="10" rx="1" fill="#ef4444"/>
    <rect x="588" y="330" width="10" height="10" rx="1" fill="#dc2626"/>
    <rect x="476" y="344" width="10" height="10" rx="1" fill="#ef4444"/>
    <rect x="490" y="344" width="10" height="10" rx="1" fill="#dc2626"/>
    <rect x="504" y="344" width="10" height="10" rx="1" fill="#b91c1c"/>
    <rect x="518" y="344" width="10" height="10" rx="1" fill="#991b1b"/>
    <rect x="532" y="344" width="10" height="10" rx="1" fill="#7f1d1d"/>
    <rect x="546" y="344" width="10" height="10" rx="1" fill="#991b1b"/>
    <rect x="560" y="344" width="10" height="10" rx="1" fill="#b91c1c"/>
    <rect x="574" y="344" width="10" height="10" rx="1" fill="#dc2626"/>
    <rect x="588" y="344" width="10" height="10" rx="1" fill="#ef4444"/>
  </g>
  <text x="552" y="368" text-anchor="middle" font-size="8" fill="#f87171">W_V ∈ ℝ^{d×d_v}</text>

  <!-- ── SCALED DOT-PRODUCT ATTENTION  (right panel) ── -->
  <rect x="652" y="286" width="216" height="192" rx="8" fill="#080820" stroke="#334155" stroke-width="1"/>
  <text x="760" y="304" text-anchor="middle" font-size="9" fill="#94a3b8" letter-spacing="1">SCALED DOT-PRODUCT</text>

  <!-- QK^T -->
  <rect x="668" y="310" width="180" height="26" rx="4" fill="#1e1b4b" stroke="#4f46e5" stroke-width="0.8"/>
  <text x="758" y="327" text-anchor="middle" font-size="9" fill="#a5b4fc">QKᵀ  ·  1/√d_k</text>

  <!-- Softmax heatmap — 6×6 attention matrix -->
  <g transform="translate(668, 342)">
    <rect width="180" height="20" rx="3" fill="#0f172a" stroke="#334155" stroke-width="0.5"/>
    <text x="90" y="14" text-anchor="middle" font-size="8" fill="#64748b">softmax( · )  →  attention weights</text>
  </g>
  <!-- Attention matrix cells — 7×7 simulated heatmap -->
  <g transform="translate(668, 368)">
    <!-- Row 0 (CLS attending to all) -->
    <rect x="0"   y="0"  width="24" height="12" rx="1" fill="#312e81" opacity="0.9"/>
    <rect x="26"  y="0"  width="24" height="12" rx="1" fill="#3730a3" opacity="0.8"/>
    <rect x="52"  y="0"  width="24" height="12" rx="1" fill="#4338ca" opacity="0.7"/>
    <rect x="78"  y="0"  width="24" height="12" rx="1" fill="#4f46e5" opacity="0.75"/>
    <rect x="104" y="0"  width="24" height="12" rx="1" fill="#6366f1" opacity="0.85"/>
    <rect x="130" y="0"  width="24" height="12" rx="1" fill="#818cf8" opacity="0.95"/>
    <rect x="156" y="0"  width="24" height="12" rx="1" fill="#a5b4fc" opacity="1"><animate attributeName="opacity" values="1;0.6;1" dur="2s" repeatCount="indefinite"/></rect>
    <!-- Row 1 -->
    <rect x="0"   y="14" width="24" height="12" rx="1" fill="#1e1b4b" opacity="0.6"/>
    <rect x="26"  y="14" width="24" height="12" rx="1" fill="#7c3aed" opacity="0.9"/>
    <rect x="52"  y="14" width="24" height="12" rx="1" fill="#6d28d9" opacity="0.7"/>
    <rect x="78"  y="14" width="24" height="12" rx="1" fill="#5b21b6" opacity="0.5"/>
    <rect x="104" y="14" width="24" height="12" rx="1" fill="#4c1d95" opacity="0.4"/>
    <rect x="130" y="14" width="24" height="12" rx="1" fill="#3b0764" opacity="0.35"/>
    <rect x="156" y="14" width="24" height="12" rx="1" fill="#2e1065" opacity="0.3"/>
    <!-- Row 2 -->
    <rect x="0"   y="28" width="24" height="12" rx="1" fill="#1e1b4b" opacity="0.3"/>
    <rect x="26"  y="28" width="24" height="12" rx="1" fill="#2563eb" opacity="0.4"/>
    <rect x="52"  y="28" width="24" height="12" rx="1" fill="#3b82f6" opacity="0.9"><animate attributeName="opacity" values="0.9;0.5;0.9" dur="2.5s" repeatCount="indefinite"/></rect>
    <rect x="78"  y="28" width="24" height="12" rx="1" fill="#2563eb" opacity="0.7"/>
    <rect x="104" y="28" width="24" height="12" rx="1" fill="#1d4ed8" opacity="0.5"/>
    <rect x="130" y="28" width="24" height="12" rx="1" fill="#1e40af" opacity="0.3"/>
    <rect x="156" y="28" width="24" height="12" rx="1" fill="#1e3a8a" opacity="0.25"/>
    <!-- Row 3 -->
    <rect x="0"   y="42" width="24" height="12" rx="1" fill="#14532d" opacity="0.4"/>
    <rect x="26"  y="42" width="24" height="12" rx="1" fill="#166534" opacity="0.5"/>
    <rect x="52"  y="42" width="24" height="12" rx="1" fill="#16a34a" opacity="0.6"/>
    <rect x="78"  y="42" width="24" height="12" rx="1" fill="#22c55e" opacity="0.9"/>
    <rect x="104" y="42" width="24" height="12" rx="1" fill="#16a34a" opacity="0.65"/>
    <rect x="130" y="42" width="24" height="12" rx="1" fill="#15803d" opacity="0.45"/>
    <rect x="156" y="42" width="24" height="12" rx="1" fill="#166534" opacity="0.3"/>
    <!-- Row 4 -->
    <rect x="0"   y="56" width="24" height="12" rx="1" fill="#78350f" opacity="0.3"/>
    <rect x="26"  y="56" width="24" height="12" rx="1" fill="#92400e" opacity="0.4"/>
    <rect x="52"  y="56" width="24" height="12" rx="1" fill="#b45309" opacity="0.5"/>
    <rect x="78"  y="56" width="24" height="12" rx="1" fill="#d97706" opacity="0.65"/>
    <rect x="104" y="56" width="24" height="12" rx="1" fill="#f59e0b" opacity="0.9"><animate attributeName="opacity" values="0.9;0.5;0.9" dur="3s" begin="1s" repeatCount="indefinite"/></rect>
    <rect x="130" y="56" width="24" height="12" rx="1" fill="#d97706" opacity="0.6"/>
    <rect x="156" y="56" width="24" height="12" rx="1" fill="#b45309" opacity="0.35"/>
    <!-- Color scale legend -->
    <text x="0"   y="82" font-size="7" fill="#475569">low</text>
    <rect x="18"  y="74" width="144" height="6" rx="3" fill="url(#softmax-g)"/>
    <text x="170" y="82" font-size="7" fill="#94a3b8">high</text>
    <text x="90"  y="92" text-anchor="middle" font-size="7.5" fill="#64748b">Attention(Q,K,V) = softmax(QKᵀ/√d_k)V</text>
  </g>

  <!-- Arrows from Q,K,V to the softmax block -->
  <line x1="268" y1="322" x2="652" y2="340" stroke="#3b82f6" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arr-cyan)" opacity="0.6"/>
  <line x1="452" y1="322" x2="660" y2="340" stroke="#10b981" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arr-green)" opacity="0.6"/>
  <line x1="636" y1="322" x2="668" y2="340" stroke="#ef4444" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arr-orange)" opacity="0.6"/>

  <!-- ── ATTENTION HEADS VISUAL  (row under Q/K/V) ── -->
  <text x="100" y="392" font-size="9" fill="#475569" letter-spacing="1">6 PARALLEL ATTENTION HEADS</text>
  <g>
    <!-- Head bubbles, 6 of them -->
    <ellipse cx="134" cy="422" rx="26" ry="18" fill="#1e1b4b" stroke="#6366f1" stroke-width="1.5" filter="url(#glow-sm)">
      <animate attributeName="ry" values="18;22;18" dur="3s" repeatCount="indefinite"/>
    </ellipse>
    <text x="134" y="426" text-anchor="middle" font-size="9" fill="#a5b4fc">h₁</text>

    <ellipse cx="196" cy="422" rx="26" ry="18" fill="#1e1b4b" stroke="#818cf8" stroke-width="1.5" filter="url(#glow-sm)">
      <animate attributeName="ry" values="18;22;18" dur="3s" begin="0.5s" repeatCount="indefinite"/>
    </ellipse>
    <text x="196" y="426" text-anchor="middle" font-size="9" fill="#a5b4fc">h₂</text>

    <ellipse cx="258" cy="422" rx="26" ry="18" fill="#1e1b4b" stroke="#7c3aed" stroke-width="1.5" filter="url(#glow-sm)">
      <animate attributeName="ry" values="18;22;18" dur="3s" begin="1s" repeatCount="indefinite"/>
    </ellipse>
    <text x="258" y="426" text-anchor="middle" font-size="9" fill="#a5b4fc">h₃</text>

    <ellipse cx="320" cy="422" rx="26" ry="18" fill="#1e1b4b" stroke="#6d28d9" stroke-width="1.5" filter="url(#glow-sm)">
      <animate attributeName="ry" values="18;22;18" dur="3s" begin="1.5s" repeatCount="indefinite"/>
    </ellipse>
    <text x="320" y="426" text-anchor="middle" font-size="9" fill="#a5b4fc">h₄</text>

    <ellipse cx="382" cy="422" rx="26" ry="18" fill="#1e1b4b" stroke="#5b21b6" stroke-width="1.5" filter="url(#glow-sm)">
      <animate attributeName="ry" values="18;22;18" dur="3s" begin="2s" repeatCount="indefinite"/>
    </ellipse>
    <text x="382" y="426" text-anchor="middle" font-size="9" fill="#a5b4fc">h₅</text>

    <ellipse cx="444" cy="422" rx="26" ry="18" fill="#1e1b4b" stroke="#4c1d95" stroke-width="1.5" filter="url(#glow-sm)">
      <animate attributeName="ry" values="18;22;18" dur="3s" begin="2.5s" repeatCount="indefinite"/>
    </ellipse>
    <text x="444" y="426" text-anchor="middle" font-size="9" fill="#a5b4fc">h₆</text>

    <!-- Concat arrow -->
    <line x1="470" y1="422" x2="508" y2="422" stroke="#4f46e5" stroke-width="1.2" marker-end="url(#arr-purple)"/>
    <rect x="512" y="410" width="120" height="24" rx="6" fill="#1e1b4b" stroke="#4f46e5" stroke-width="1"/>
    <text x="572" y="426" text-anchor="middle" font-size="9" fill="#818cf8">Concat → W_O</text>
  </g>

  <!-- Output projection label -->
  <text x="100" y="462" font-size="8" fill="#475569">Multi-Head Output =  Concat(head₁,...,headₕ) · W_O     where  headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)</text>

  <!-- ── RESIDUAL ADD 1 ── -->
  <!-- Residual bypass curve -->
  <path d="M64,240 C40,340 40,440 64,490" stroke="#06b6d4" stroke-width="1.5" fill="none" stroke-dasharray="5,3" opacity="0.7"/>
  <circle cx="64" cy="490" r="10" fill="#083344" stroke="#06b6d4" stroke-width="1.5" filter="url(#glow-sm)"/>
  <text x="64" y="494" text-anchor="middle" font-size="10" fill="#06b6d4" font-weight="bold">+</text>
  <text x="30" y="490" font-size="8" fill="#0e7490">Add</text>
  <line x1="640" y1="490" x2="74" y2="490" stroke="#06b6d4" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/>

  <!-- ══════════════════════════════════════════
       LAYER NORM 2  (y: 502–536)
  ══════════════════════════════════════════ -->
  <line x1="450" y1="492" x2="450" y2="502" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arr-purple)"/>
  <rect x="82" y="502" width="800" height="38" rx="8" fill="#0f0a2a" stroke="#7c3aed" stroke-width="1.2"/>
  <g>
    <rect x="90" y="510" width="12" height="22" rx="2" fill="#4c1d95"/>
    <rect x="106" y="510" width="12" height="22" rx="2" fill="#5b21b6"/>
    <rect x="122" y="510" width="12" height="22" rx="2" fill="#6d28d9"/>
    <rect x="138" y="510" width="12" height="22" rx="2" fill="#7c3aed"/>
    <rect x="154" y="510" width="12" height="22" rx="2" fill="#8b5cf6"/>
    <rect x="170" y="510" width="12" height="22" rx="2" fill="#7c3aed"/>
    <rect x="186" y="510" width="12" height="22" rx="2" fill="#6d28d9"/>
    <rect x="202" y="510" width="12" height="22" rx="2" fill="#5b21b6"/>
  </g>
  <text x="440" y="525" text-anchor="middle" font-size="11" fill="#c4b5fd" font-weight="bold" letter-spacing="2">LAYER NORM 2</text>
  <text x="750" y="525" text-anchor="middle" font-size="9" fill="#6d28d9">x̂ = (x − μ) / σ · γ + β</text>

  <!-- flow down -->
  <line x1="450" y1="540" x2="450" y2="556" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>

  <!-- ══════════════════════════════════════════
       FEED-FORWARD NETWORK  (y: 560–664)
  ══════════════════════════════════════════ -->
  <rect x="82" y="560" width="800" height="108" rx="10" fill="url(#ffn-g)" stroke="#059669" stroke-width="1.5"/>
  <text x="440" y="580" text-anchor="middle" font-size="12" fill="#6ee7b7" font-weight="bold" letter-spacing="3">FEED-FORWARD NETWORK</text>

  <!-- FFN: Linear1 -> GELU -> Linear2 -->
  <!-- Linear 1 box -->
  <rect x="100" y="590" width="180" height="58" rx="7" fill="#064e3b" stroke="#10b981" stroke-width="1"/>
  <text x="190" y="612" text-anchor="middle" font-size="10" fill="#a7f3d0" font-weight="bold">Linear₁</text>
  <text x="190" y="628" text-anchor="middle" font-size="8" fill="#6ee7b7">W₁ ∈ ℝ^{d×4d}</text>
  <text x="190" y="641" text-anchor="middle" font-size="8" fill="#34d399">d→ 4·d  (expand)</text>

  <!-- Arrow -->
  <line x1="280" y1="619" x2="314" y2="619" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <!-- GELU -->
  <rect x="318" y="590" width="128" height="58" rx="7" fill="#065f46" stroke="#34d399" stroke-width="1">
    <animate attributeName="stroke-opacity" values="1;0.3;1" dur="2.5s" repeatCount="indefinite"/>
  </rect>
  <text x="382" y="612" text-anchor="middle" font-size="10" fill="#6ee7b7" font-weight="bold">GELU</text>
  <text x="382" y="628" text-anchor="middle" font-size="8" fill="#a7f3d0">σ(x)·x</text>
  <text x="382" y="641" text-anchor="middle" font-size="8" fill="#4ade80">non-linear</text>
  <!-- Arrow -->
  <line x1="446" y1="619" x2="480" y2="619" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <!-- Linear 2 -->
  <rect x="484" y="590" width="180" height="58" rx="7" fill="#064e3b" stroke="#10b981" stroke-width="1"/>
  <text x="574" y="612" text-anchor="middle" font-size="10" fill="#a7f3d0" font-weight="bold">Linear₂</text>
  <text x="574" y="628" text-anchor="middle" font-size="8" fill="#6ee7b7">W₂ ∈ ℝ^{4d×d}</text>
  <text x="574" y="641" text-anchor="middle" font-size="8" fill="#34d399">4d→ d  (project)</text>

  <!-- Formula -->
  <text x="440" y="660" text-anchor="middle" font-size="9" fill="#475569">FFN(x) = GELU(xW₁ + b₁)W₂ + b₂    ·    d_ff = 4 × d_model = 1536</text>

  <!-- ── RESIDUAL ADD 2 ── -->
  <path d="M64,502 C40,570 40,660 64,674" stroke="#a855f7" stroke-width="1.5" fill="none" stroke-dasharray="5,3" opacity="0.7"/>
  <circle cx="64" cy="674" r="10" fill="#1e0a40" stroke="#a855f7" stroke-width="1.5" filter="url(#glow-sm)"/>
  <text x="64" y="678" text-anchor="middle" font-size="10" fill="#a855f7" font-weight="bold">+</text>
  <text x="30" y="674" font-size="8" fill="#7c3aed">Add</text>

  <!-- ══════════════════════════════════════════
       BLOCK OUTPUT + REPEAT LABEL  (y: 690–752)
  ══════════════════════════════════════════ -->
  <line x1="450" y1="668" x2="450" y2="690" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr-white)"/>

  <rect x="82" y="692" width="800" height="38" rx="8" fill="#0d1424" stroke="#334155" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="440" y="716" text-anchor="middle" font-size="11" fill="#64748b" letter-spacing="2">↑  REPEAT  ×L  LAYERS  ↑    (L = 12 for ViT-S/16)</text>

  <!-- ══════════════════════════════════════════
       FLOW ARROW OUT OF TRANSFORMER BLOCK
  ══════════════════════════════════════════ -->
  <line x1="450" y1="754" x2="450" y2="775" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-white)"/>

  <!-- ══════════════════════════════════════════
       FORK: JEPA PREDICTOR  +  TTA  (y: 778–890)
  ══════════════════════════════════════════ -->
  <!-- Fork line -->
  <line x1="250" y1="778" x2="650" y2="778" stroke="#334155" stroke-width="1.5"/>
  <line x1="250" y1="778" x2="250" y2="800" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <line x1="650" y1="778" x2="650" y2="800" stroke="#f97316" stroke-width="1.5" marker-end="url(#arr-orange)"/>

  <!-- ── JEPA PREDICTOR block ── -->
  <rect x="82" y="800" width="336" height="118" rx="10" fill="url(#jepa-g)" stroke="#059669" stroke-width="1.5"/>
  <text x="250" y="820" text-anchor="middle" font-size="11" fill="#34d399" font-weight="bold" letter-spacing="2">JEPA PREDICTOR</text>

  <!-- Context / Target split -->
  <rect x="96" y="830" width="140" height="52" rx="6" fill="#053d2a" stroke="#10b981" stroke-width="1"/>
  <text x="166" y="849" text-anchor="middle" font-size="9" fill="#6ee7b7" font-weight="bold">Context Encoder</text>
  <!-- visible patches -->
  <g>
    <rect x="104" y="856" width="18" height="18" rx="2" fill="#10b981" opacity="0.7"/>
    <rect x="126" y="856" width="18" height="18" rx="2" fill="#10b981" opacity="0.6"/>
    <rect x="148" y="856" width="18" height="18" rx="2" fill="none" stroke="#10b981" stroke-dasharray="3,2"/>
    <rect x="170" y="856" width="18" height="18" rx="2" fill="none" stroke="#10b981" stroke-dasharray="3,2"/>
    <rect x="192" y="856" width="18" height="18" rx="2" fill="#10b981" opacity="0.5"/>
    <rect x="214" y="856" width="18" height="18" rx="2" fill="none" stroke="#10b981" stroke-dasharray="3,2"/>
  </g>

  <!-- Arrow between -->
  <line x1="236" y1="856" x2="252" y2="856" stroke="#10b981" stroke-width="1.2" marker-end="url(#arr-green)"/>

  <!-- Predictor box -->
  <rect x="256" y="830" width="148" height="52" rx="6" fill="#064e3b" stroke="#34d399" stroke-width="1">
    <animate attributeName="stroke-opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>
  </rect>
  <text x="330" y="849" text-anchor="middle" font-size="9" fill="#a7f3d0" font-weight="bold">Target Prediction</text>
  <text x="330" y="864" text-anchor="middle" font-size="8" fill="#6ee7b7">ẑ = f(z_context)</text>
  <text x="330" y="876" text-anchor="middle" font-size="8" fill="#34d399">L = ||ẑ − z_target||²</text>

  <text x="250" y="908" text-anchor="middle" font-size="8" fill="#475569">predict embeddings, not pixels  ·  no decoder needed</text>

  <!-- ── TTA block ── -->
  <rect x="482" y="800" width="336" height="118" rx="10" fill="url(#tta-g)" stroke="#ea580c" stroke-width="1.5"/>
  <text x="650" y="820" text-anchor="middle" font-size="11" fill="#fb923c" font-weight="bold" letter-spacing="2">TEST-TIME ADAPTATION</text>

  <!-- 3 TTA sub-modules -->
  <rect x="496" y="830" width="96" height="54" rx="6" fill="#3b1108" stroke="#f97316" stroke-width="0.8"/>
  <text x="544" y="850" text-anchor="middle" font-size="8" fill="#fdba74" font-weight="bold">Entropy</text>
  <text x="544" y="863" text-anchor="middle" font-size="8" fill="#fb923c">Min.</text>
  <text x="544" y="876" text-anchor="middle" font-size="7" fill="#92400e">H(p)→min</text>

  <rect x="602" y="830" width="96" height="54" rx="6" fill="#3b1108" stroke="#ea580c" stroke-width="0.8"/>
  <text x="650" y="850" text-anchor="middle" font-size="8" fill="#fdba74" font-weight="bold">BatchNorm</text>
  <text x="650" y="863" text-anchor="middle" font-size="8" fill="#fb923c">Adapt</text>
  <text x="650" y="876" text-anchor="middle" font-size="7" fill="#92400e">μ,σ update</text>

  <rect x="708" y="830" width="96" height="54" rx="6" fill="#3b1108" stroke="#c2410c" stroke-width="0.8"/>
  <text x="756" y="850" text-anchor="middle" font-size="8" fill="#fdba74" font-weight="bold">Aug.</text>
  <text x="756" y="863" text-anchor="middle" font-size="8" fill="#fb923c">Consist.</text>
  <text x="756" y="876" text-anchor="middle" font-size="7" fill="#92400e">no labels</text>

  <text x="650" y="908" text-anchor="middle" font-size="8" fill="#475569">adapts at inference time  ·  zero labels  ·  ImageNet-C/R</text>

  <!-- ══════════════════════════════════════════
       OUTPUT: CLASSIFIER  (y: 920–990)
  ══════════════════════════════════════════ -->
  <line x1="250" y1="918" x2="250" y2="938" stroke="#94a3b8" stroke-width="1.2" marker-end="url(#arr-white)"/>
  <line x1="650" y1="918" x2="650" y2="938" stroke="#94a3b8" stroke-width="1.2" marker-end="url(#arr-white)"/>
  <line x1="250" y1="938" x2="650" y2="938" stroke="#334155" stroke-width="1.2"/>
  <line x1="450" y1="938" x2="450" y2="948" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-white)"/>

  <rect x="224" y="950" width="452" height="52" rx="10" fill="url(#out-g)" stroke="#6366f1" stroke-width="1.5" filter="url(#glow-b)"/>
  <text x="450" y="972" text-anchor="middle" font-size="12" fill="#a5b4fc" font-weight="bold" letter-spacing="3">LINEAR CLASSIFIER  (linear probe)</text>
  <text x="450" y="990" text-anchor="middle" font-size="9" fill="#4f46e5">W ∈ ℝ^{d×C}  ·  softmax  →  class probabilities</text>

  <!-- ══════════════════════════════════════════
       LEGEND  (right rail, y: 800-920)
  ══════════════════════════════════════════ -->

  <!-- ══════════════════════════════════════════
       ANIMATED DATA-FLOW PARTICLES
  ══════════════════════════════════════════ -->
  <!-- Cyan particle flowing down the main stream -->
  <circle r="4" fill="#06b6d4" filter="url(#glow-cyan)">
    <animateMotion dur="6s" repeatCount="indefinite"
      path="M 450,148 L 450,196 L 450,260 L 450,502 L 450,560 L 450,668 L 450,754 L 450,778"/>
    <animate attributeName="opacity" values="0;1;1;1;1;0" dur="6s" repeatCount="indefinite"/>
  </circle>

  <!-- Purple particle, slight delay -->
  <circle r="3" fill="#a855f7" filter="url(#glow-sm)">
    <animateMotion dur="6s" begin="2s" repeatCount="indefinite"
      path="M 450,148 L 450,196 L 450,260 L 450,502 L 450,560 L 450,668 L 450,754 L 450,778"/>
    <animate attributeName="opacity" values="0;0.8;0.8;0.8;0.8;0" dur="6s" begin="2s" repeatCount="indefinite"/>
  </circle>

  <!-- Green particle on residual path -->
  <circle r="3" fill="#10b981">
    <animateMotion dur="4s" begin="1s" repeatCount="indefinite"
      path="M 64,240 C 40,340 40,440 64,490 L 64,674"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0" dur="4s" begin="1s" repeatCount="indefinite"/>
  </circle>
</svg>
