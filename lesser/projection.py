"""Two-sided random projections of the output-layer gradient.

The gradient sum_t w_t r_t h_t^T (V x d) is never formed: each token's residual r_t and hidden state
h_t are projected first, and the projected outer products are summed. With Rademacher maps
P_v (V x k_v) and P_h (d x k_h), the feature is sum_t w_t (P_h^T h_t)(P_v^T r_t)^T.

The paper's settings drew the random maps and laid out the result differently. Each preset below
reproduces one setting exactly, so released features match the ones behind the reported numbers:

* ``SFT_PAPER``: hidden map d -> 128 from a CPU generator seeded 0, vocabulary map V -> 64 from a
  separate generator seeded 1; result laid out as (128, 64) and flattened (8,192 entries).
* ``RL_PAPER``: one CPU generator seeded 0 draws the vocabulary map V -> 64 first, then the hidden
  map d -> 128; result laid out as (64, 128). The residual projection is computed as
  softmax @ P_v - P_v[y], as in the RL scorer.

GRACE used a dense TRAK projection of the exact gradient instead; see ``grace/`` in this repo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TwoSidedConfig:
    k_hidden: int
    k_vocab: int
    seed: int
    draw: str      # "separate_generators" (SFT) or "single_generator_vocab_first" (RL)
    layout: str    # "hidden_vocab" -> (k_hidden, k_vocab); "vocab_hidden" -> (k_vocab, k_hidden)


SFT_PAPER = TwoSidedConfig(k_hidden=128, k_vocab=64, seed=0, draw="separate_generators", layout="hidden_vocab")
RL_PAPER = TwoSidedConfig(k_hidden=128, k_vocab=64, seed=0, draw="single_generator_vocab_first", layout="vocab_hidden")


def _rademacher(bits: torch.Tensor, k: int) -> torch.Tensor:
    return (bits.to(torch.float32) * 2.0 - 1.0) / math.sqrt(k)


class TwoSidedProjection:
    """Rademacher maps for one (vocabulary size, hidden size) and one preset."""

    def __init__(self, vocab_size: int, hidden_size: int, config: TwoSidedConfig = SFT_PAPER,
                 device: torch.device | str = "cpu"):
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        self.config = config
        kh, kv = config.k_hidden, config.k_vocab
        if config.draw == "separate_generators":
            gh = torch.Generator(device="cpu").manual_seed(config.seed)
            hbits = torch.randint(0, 2, (hidden_size, kh), generator=gh, dtype=torch.int8)
            gv = torch.Generator(device="cpu").manual_seed(config.seed + 1)
            vbits = torch.randint(0, 2, (vocab_size, kv), generator=gv, dtype=torch.int8)
        elif config.draw == "single_generator_vocab_first":
            g = torch.Generator(device="cpu").manual_seed(config.seed)
            vbits = torch.randint(0, 2, (vocab_size, kv), generator=g, dtype=torch.int8)
            hbits = torch.randint(0, 2, (hidden_size, kh), generator=g, dtype=torch.int8)
        else:
            raise ValueError(f"unknown draw scheme {config.draw!r}")
        self.P_h = _rademacher(hbits, kh).to(device)
        self.P_v = _rademacher(vbits, kv).to(device)

    @property
    def dim(self) -> int:
        return self.config.k_hidden * self.config.k_vocab

    def _combine(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (T, k_hidden) projected hidden states; B: (T, k_vocab) projected (weighted) residuals
        G = A.T @ B if self.config.layout == "hidden_vocab" else B.T @ A
        return G.reshape(-1)

    def project(self, h: torch.Tensor, r: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Project sum_t w_t r_t h_t^T from per-token factors h (T, d) and r (T, V)."""
        A = h.float() @ self.P_h
        B = r.float() @ self.P_v
        if weights is not None:
            B = B * weights.unsqueeze(-1)
        return self._combine(A, B)

    def project_from_probs(self, h: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor,
                           weights: torch.Tensor | None = None) -> torch.Tensor:
        """RL-scorer form: projected residual computed as probs @ P_v - P_v[y] without forming r."""
        A = h.float() @ self.P_h
        B = probs.float() @ self.P_v - self.P_v[labels]
        if weights is not None:
            B = B * weights.unsqueeze(-1)
        return self._combine(A, B)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise (or vector) L2 normalization used before cosine scoring."""
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
