"""
Aletheia Twohot Discrete Regression — DreamerV3 Style
=====================================================

Core utilities for twohot encoding / decoding of scalar values
as soft probability distributions over fixed bins.

Reference: Hafner et al., "Mastering Diverse Domains through World Models"
           (DreamerV3, 2023), Section B.3.

The key idea: instead of regressing a single scalar value, the critic
outputs logits over a set of bins.  The target is encoded as a "twohot"
vector — a soft label where probability mass is shared between the two
bins adjacent to the target value.

Benefits:
  - Captures multi-modal / asymmetric value distributions
  - More stable gradients than scalar MSE on extreme values
  - Works naturally with symlog-space bins for scale-free learning
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .aletheia_foundation import SymlogLayer

Tensor = torch.Tensor
_SAFE_SYMLOG = SymlogLayer()


# ======================================================================
# Bin Construction
# ======================================================================

def symlog_bins(
    num_bins: int = 255,
    vmin: float = -20.0,
    vmax: float = 20.0,
) -> Tensor:
    """Create uniformly spaced bins in symlog space.

    Parameters
    ----------
    num_bins : int
        Number of bins (DreamerV3 default: 255).
    vmin, vmax : float
        Range in symlog space (DreamerV3 default: [-20, 20]).

    Returns
    -------
    Tensor, shape ``(num_bins,)``
        Bin centers in symlog space.
    """
    return torch.linspace(vmin, vmax, num_bins)


# ======================================================================
# Twohot Encoding
# ======================================================================

def twohot_encode(
    value: Tensor,
    bins: Tensor,
) -> Tensor:
    """Encode scalar values as twohot soft labels.

    Each value is distributed between the two adjacent bins
    proportionally to its distance from each bin center.

    Parameters
    ----------
    value : Tensor, shape ``(...)``
        Values in symlog space to encode.
    bins : Tensor, shape ``(B,)``
        Sorted bin centers in symlog space.

    Returns
    -------
    Tensor, shape ``(..., B)``
        Twohot probability distribution over bins.
    """
    # Clamp value to bin range
    value = value.clamp(bins[0], bins[-1])

    # Find the bin index just below the value
    # bins: sorted ascending, shape (B,)
    # value: (...), expand for comparison
    below = (bins.unsqueeze(0) <= value.unsqueeze(-1)).sum(dim=-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1

    # Compute interpolation weight
    below_val = bins[below]
    above_val = bins[above]
    weight = (value - below_val) / (above_val - below_val + 1e-8)
    weight = weight.clamp(0.0, 1.0)

    # Build twohot: put (1 - weight) on below, weight on above
    target = torch.zeros(*value.shape, len(bins), device=value.device, dtype=value.dtype)
    target.scatter_(-1, below.unsqueeze(-1), (1.0 - weight).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))

    return target


# ======================================================================
# Twohot Decoding
# ======================================================================

def twohot_decode(
    probs: Tensor,
    bins: Tensor,
) -> Tensor:
    """Decode twohot probabilities to scalar values.

    Computes the expected value: sum(probs * bins).

    Parameters
    ----------
    probs : Tensor, shape ``(..., B)``
        Probability distribution over bins.
    bins : Tensor, shape ``(B,)``
        Sorted bin centers in symlog space.

    Returns
    -------
    Tensor, shape ``(...)``
        Decoded scalar values in symlog space.
    """
    return (probs * bins).sum(dim=-1)


# ======================================================================
# Loss Function
# ======================================================================

def twohot_loss(
    logits: Tensor,
    target: Tensor,
    bins: Tensor,
) -> Tensor:
    """Cross-entropy loss between predicted logits and twohot targets.

    Parameters
    ----------
    logits : Tensor, shape ``(..., B)``
        Raw (unnormalized) logits from the critic head.
    target : Tensor, shape ``(...)``
        Target values in **real** (not symlog) space.
        Will be converted to symlog internally.
    bins : Tensor, shape ``(B,)``
        Bin centers in symlog space.

    Returns
    -------
    Tensor, shape ``(...)``
        Per-element cross-entropy loss (not reduced).
    """
    # Convert target to symlog space
    target_symlog = _SAFE_SYMLOG.symlog(target)

    # Encode as twohot
    twohot_target = twohot_encode(target_symlog, bins)

    # Cross-entropy: -sum(target * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(twohot_target * log_probs).sum(dim=-1)

    return loss


# ======================================================================
# TwohotHead — drop-in replacement for scalar value heads
# ======================================================================

class TwohotHead(nn.Module):
    """Twohot discrete regression head.

    Drop-in replacement for a scalar output head.  Instead of outputting
    a single value, outputs logits over bins and decodes via expectation.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    num_bins : int
        Number of bins (default: 255).
    vmin, vmax : float
        Range in symlog space (default: [-20, 20]).
    """

    def __init__(
        self,
        in_dim: int,
        num_bins: int = 255,
        vmin: float = -20.0,
        vmax: float = 20.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.head = nn.Linear(in_dim, num_bins)

        # Initialize with small weights for stable start
        nn.init.zeros_(self.head.bias)
        nn.init.uniform_(self.head.weight, -0.01, 0.01)

        # Register bins as buffer (moves with model to GPU, not a parameter)
        self.register_buffer("bins", symlog_bins(num_bins, vmin, vmax))

    def forward(self, x: Tensor) -> Tensor:
        """Return logits, shape ``(..., num_bins)``."""
        return self.head(x)

    def decode(self, logits: Tensor) -> Tensor:
        """Decode logits → scalar value in symlog space."""
        probs = F.softmax(logits, dim=-1)
        return twohot_decode(probs, self.bins)

    def decode_real(self, logits: Tensor) -> Tensor:
        """Decode logits → scalar value in real space."""
        val_sym = self.decode(logits)
        return torch.sign(val_sym) * (torch.exp(torch.abs(val_sym)) - 1.0)

    def compute_loss(self, logits: Tensor, target_real: Tensor) -> Tensor:
        """Compute per-element cross-entropy loss.

        Parameters
        ----------
        logits : Tensor, shape ``(..., num_bins)``
        target_real : Tensor, shape ``(...)``
            Target values in **real** (not symlog) space.

        Returns
        -------
        Tensor, shape ``(...)``
            Per-element loss (unreduced).
        """
        return twohot_loss(logits, target_real, self.bins)
