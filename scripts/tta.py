"""
Test-Time Adaptation (TTA) for JEPA-RobustViT.

Implements entropy minimisation over LayerNorm affine parameters (γ, β).

Design rationale:
  - TENT (Wang et al., 2021) adapts BatchNorm statistics at test time.
    Vision Transformers use LayerNorm, not BatchNorm — TENT cannot be
    applied directly to ViT architectures.
  - This module targets LayerNorm γ (weight) and β (bias) parameters,
    which control the scale and shift of normalised activations. These
    are the minimal set of parameters that can adapt the model's internal
    representation distribution to match the test domain.
  - No labels are required. The adaptation signal is the entropy of the
    model's own softmax predictions: high entropy = uncertain = bad.
    Minimising entropy encourages the model to make confident predictions
    on the test domain.
  - The backbone weights, attention weights, and FFN weights are all frozen.
    Only LayerNorm affine parameters are updated.

Reference:
    Wang et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization."
    ICLR 2021. https://arxiv.org/abs/2006.10726

    This implementation extends the TENT idea to LayerNorm following:
    Lim et al. "TTT++: When Does Self-Supervised Test-Time Training
    Fail or Thrive?" NeurIPS 2021.
"""

import copy
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Entropy computation
# ---------------------------------------------------------------------------

def entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample prediction entropy from logits.

    H(p) = -sum_c p_c * log(p_c)

    Uses the numerically stable log-softmax formulation:
    H(p) = -sum_c softmax(x)_c * log_softmax(x)_c

    Args:
        logits: (B, C) unnormalised class logits

    Returns:
        entropy: (B,) per-sample entropy in nats
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs     = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=1)


def mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute mean entropy over a batch. Used as the TTA loss.

    Args:
        logits: (B, C) unnormalised class logits

    Returns:
        scalar mean entropy tensor
    """
    return entropy(logits).mean()


# ---------------------------------------------------------------------------
# LayerNorm parameter collection
# ---------------------------------------------------------------------------

def collect_layernorm_params(
    model: nn.Module,
    requires_grad: bool = True,
) -> List[nn.Parameter]:
    """
    Collect all LayerNorm affine parameters (weight and bias) from a model.

    Args:
        model:         the model to inspect
        requires_grad: if True, set requires_grad=True on collected params

    Returns:
        list of LayerNorm parameter tensors (weights and biases)
    """
    params = []
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                if requires_grad:
                    module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                if requires_grad:
                    module.bias.requires_grad_(True)
                params.append(module.bias)
    return params


def freeze_non_layernorm_params(model: nn.Module) -> None:
    """
    Freeze all parameters in the model except LayerNorm affine parameters.

    After calling this function:
      - LayerNorm weight and bias: requires_grad = True
      - All other parameters:      requires_grad = False

    Args:
        model: the model to partially freeze
    """
    for name, param in model.named_parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)


# ---------------------------------------------------------------------------
# TTA wrapper
# ---------------------------------------------------------------------------

class TTAWrapper(nn.Module):
    """
    Test-Time Adaptation wrapper for a JEPAClassifier (or any model with
    LayerNorm layers).

    Wraps an existing trained model and adapts its LayerNorm affine
    parameters by minimising prediction entropy on test batches.

    Usage:
        # After training, wrap the model for TTA
        tta_model = TTAWrapper(trained_classifier, lr=1e-4, steps=1)

        # At test time, call forward() — it adapts and returns predictions
        with torch.no_grad():
            logits = tta_model(test_batch)

    Args:
        model:       trained JEPAClassifier (or compatible model)
        lr:          learning rate for LayerNorm parameter updates (1e-4)
        steps:       number of gradient steps per test batch (1)
        episodic:    if True, reset model to original weights after each
                     forward call (episodic TTA). If False, accumulate
                     adaptations across batches (continual TTA).
        entropy_threshold: if not None, only adapt on samples whose entropy
                     is above this threshold. Samples already confidently
                     predicted are excluded from the adaptation loss.
                     Set to None to adapt on all samples.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        steps: int = 1,
        episodic: bool = True,
        entropy_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.steps = steps
        self.episodic = episodic
        self.entropy_threshold = entropy_threshold

        # Deep copy the model so the original is never modified
        self.model = copy.deepcopy(model)

        # Save original LayerNorm parameters for episodic reset
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Freeze everything except LayerNorm affine parameters
        freeze_non_layernorm_params(self.model)

        # Collect trainable parameters
        self.params = collect_layernorm_params(self.model, requires_grad=True)
        assert len(self.params) > 0, (
            "No LayerNorm parameters found in model. "
            "Ensure the model contains nn.LayerNorm layers."
        )

        # Optimiser over LayerNorm parameters only
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

    def reset(self) -> None:
        """
        Reset model weights to original pre-adaptation state.
        Called automatically when episodic=True.
        """
        self.model.load_state_dict(self._original_state)
        # Re-collect params after reset (tensors are replaced)
        self.params = collect_layernorm_params(self.model, requires_grad=True)
        self.optimizer = torch.optim.Adam(
            self.params, lr=self.optimizer.param_groups[0]["lr"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with test-time adaptation.

        Steps:
          1. Run `self.steps` gradient steps minimising entropy on this batch
          2. Return final logits (with adapted LayerNorm parameters)
          3. If episodic=True, reset model weights for the next batch

        Args:
            x: (B, 3, H, W) test batch

        Returns:
            logits: (B, C) class logits after adaptation
        """
        if self.episodic:
            self.reset()

        # Adaptation steps
        for _ in range(self.steps):
            self.model.train()
            logits = self.model(x)

            if self.entropy_threshold is not None:
                # Filter: only adapt on uncertain samples
                batch_entropy = entropy(logits)
                mask = batch_entropy >= self.entropy_threshold
                if mask.sum() == 0:
                    # All samples are already confident — skip adaptation
                    break
                loss = entropy(logits[mask]).mean()
            else:
                loss = mean_entropy(logits)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Final forward pass with adapted weights
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        return logits

    @property
    def num_adaptable_params(self) -> int:
        """Number of LayerNorm parameters being adapted."""
        return sum(p.numel() for p in self.params)

    def __repr__(self) -> str:
        return (
            f"TTAWrapper("
            f"steps={self.steps}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
            f"episodic={self.episodic}, "
            f"adaptable_params={self.num_adaptable_params:,})"
        )