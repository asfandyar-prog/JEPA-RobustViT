"""
I-JEPA Predictor Head for JEPA-RobustViT.

The predictor is a narrow transformer that takes context encoder features
and target block positional embeddings, and predicts the target encoder
representations for masked regions.

Architecture follows Assran et al., CVPR 2023:
  - Input:  context features (B, N_ctx, D) + target positions (B, N_tgt, D)
  - Output: predicted target features (B, N_tgt, D)
  - Design: lightweight — fewer layers and heads than the backbone encoder

Reference:
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023.
    https://arxiv.org/abs/2301.08243
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
) -> torch.Tensor:
    """
    Generate 2D sine-cosine positional embeddings for a square patch grid.

    Args:
        embed_dim: embedding dimension (must be even)
        grid_size: number of patches along each spatial dimension
                   e.g. 14 for ViT-B/16 with 224×224 input

    Returns:
        pos_embed: (grid_size**2, embed_dim) positional embedding tensor
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sincos embedding"

    half_dim = embed_dim // 2
    positions = torch.arange(grid_size, dtype=torch.float32)
    omega = torch.arange(half_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (2 * omega / half_dim))

    # Build frequency matrix for each axis
    out_h = torch.outer(positions, omega)  # (grid_size, half_dim//2)
    out_w = torch.outer(positions, omega)

    # Interleave sin and cos
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (G, half_dim)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)

    # Combine grid
    grid_h, grid_w = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
    )
    grid_h = grid_h.flatten()  # (G*G,)
    grid_w = grid_w.flatten()

    pos_embed = torch.cat([emb_h[grid_h], emb_w[grid_w]], dim=1)  # (G*G, D)
    return pos_embed


# ---------------------------------------------------------------------------
# Predictor transformer block
# ---------------------------------------------------------------------------

class PredictorBlock(nn.Module):
    """
    Single transformer block used inside the JEPA predictor.

    Uses pre-norm (LayerNorm before attention/FFN) following ViT convention.
    The predictor uses cross-attention: queries come from target positions,
    keys/values come from context features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_queries: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_queries:   (B, N_tgt, D) — positional queries for target blocks
            context_features: (B, N_ctx, D) — encoded context patch features

        Returns:
            output: (B, N_tgt, D) — updated target queries
        """
        q = self.norm1_q(target_queries)
        kv = self.norm1_kv(context_features)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        target_queries = target_queries + attn_out

        target_queries = target_queries + self.mlp(self.norm2(target_queries))
        return target_queries


# ---------------------------------------------------------------------------
# Main predictor module
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """
    Lightweight transformer predictor for I-JEPA.

    Takes context encoder output and target patch positions, predicts
    the target encoder representations for the masked target blocks.

    Design choices following Assran et al. 2023:
      - Narrower than the backbone: 384-dim, 6 heads, 6 layers
      - Input projection: backbone_dim (768) → predictor_dim (384)
      - Output projection: predictor_dim (384) → backbone_dim (768)
      - Positional embeddings are added to target queries only

    Args:
        backbone_dim:   feature dimension of the backbone encoder (768)
        predictor_dim:  internal dimension of the predictor (384)
        num_heads:      number of attention heads (6)
        num_layers:     number of transformer blocks (6)
        mlp_ratio:      FFN hidden dim multiplier (4.0)
        dropout:        dropout rate (0.0 during pretraining)
        grid_size:      spatial grid size of backbone patches (14 for ViT-B/16)
    """

    def __init__(
        self,
        backbone_dim: int = 768,
        predictor_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        grid_size: int = 14,
    ) -> None:
        super().__init__()

        self.backbone_dim = backbone_dim
        self.predictor_dim = predictor_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size  # 196 for ViT-B/16

        # Project backbone features into predictor space
        self.input_proj = nn.Linear(backbone_dim, predictor_dim)

        # Learnable positional embeddings for all patch positions
        # These are used to construct target queries at masked positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_dim),
            requires_grad=True,
        )

        # Transformer blocks (cross-attention: target queries attend to context)
        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to backbone dimension for L2 loss with target encoder
        self.output_proj = nn.Linear(predictor_dim, backbone_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following ViT convention."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        context_features: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target encoder representations for masked patch positions.

        Args:
            context_features: (B, N_ctx, backbone_dim)
                               Output of context encoder for visible patches.
                               N_ctx = number of unmasked patches.

            target_indices:   (B, N_tgt)
                               Flat patch indices (0..195) of the target blocks.
                               N_tgt = number of masked target patches.

        Returns:
            predictions: (B, N_tgt, backbone_dim)
                         Predicted representations in backbone feature space.
                         Compared against target encoder output via L2 loss.
        """
        B, N_ctx, _ = context_features.shape
        N_tgt = target_indices.shape[1]

        # Project context features to predictor dimension
        ctx = self.input_proj(context_features)  # (B, N_ctx, predictor_dim)

        # Build target queries from positional embeddings at masked positions
        # pos_embed shape: (1, num_patches, predictor_dim)
        # Gather positions for each target index in the batch
        pos = self.pos_embed.expand(B, -1, -1)  # (B, num_patches, predictor_dim)
        idx = target_indices.unsqueeze(-1).expand(
            -1, -1, self.predictor_dim
        )  # (B, N_tgt, predictor_dim)
        target_queries = torch.gather(pos, dim=1, index=idx)  # (B, N_tgt, predictor_dim)

        # Run cross-attention blocks
        x = target_queries
        for block in self.blocks:
            x = block(target_queries=x, context_features=ctx)

        x = self.norm(x)

        # Project back to backbone dimension
        predictions = self.output_proj(x)  # (B, N_tgt, backbone_dim)
        return predictions