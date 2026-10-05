"""Output-layer gradient of a weighted token-level loss (Proposition 1 of the paper).

For a loss L_p = -sum_t alpha_t log pi(y_t | x, y_<t), the gradient with respect to a readout copy
W_out of the final vocabulary projection W is

    g_p = sum_t alpha_t (softmax(W h_t) - e_{y_t}) h_t^T,

where h_t is the final (post-norm) hidden state that predicts y_t. Both factors come from one
forward pass, so no backpropagation is needed.

This module returns the per-token factors (h_t, r_t) with r_t = softmax(W h_t) - e_{y_t}, and the
exact gradient g_p. Projections of g_p live in ``lesser.projection``.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -100


def supervised_positions(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """Next-token convention: the hidden state at position p - 1 predicts the label at position p.

    Returns (source positions, target ids). A supervised label at position 0 has no preceding
    hidden state and is dropped.
    """
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {tuple(labels.shape)}")
    pos = (labels != ignore_index).nonzero(as_tuple=False).flatten()
    pos = pos[pos >= 1]
    return pos - 1, labels[pos]


def token_factors(hidden: torch.Tensor, W: torch.Tensor, labels: torch.Tensor,
                  ignore_index: int = IGNORE_INDEX):
    """Per-token factors from final hidden states and the readout matrix, in fp32.

    hidden: (T, d) final hidden states of one sequence; W: (V, d) readout; labels: (T,) with
    ``ignore_index`` on unsupervised positions. Returns h (T_r, d) and r (T_r, V) with
    r_t = softmax(W h_t) - e_{y_t}. The logits are recomputed in fp32 from h and W (the SFT
    extraction convention); use :func:`token_factors_from_logits` to reuse model logits instead.
    """
    src, y = supervised_positions(labels, ignore_index)
    if src.numel() == 0:
        return hidden.new_zeros((0, hidden.shape[1]), dtype=torch.float32), \
            hidden.new_zeros((0, W.shape[0]), dtype=torch.float32)
    h = hidden[src].float()
    r = torch.softmax(h @ W.float().T, dim=-1)
    r[torch.arange(r.shape[0], device=r.device), y] -= 1.0
    return h, r


def token_factors_from_logits(hidden: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor,
                              ignore_index: int = IGNORE_INDEX):
    """Same as :func:`token_factors` but takes the model's own logits (T, V) (GRACE and RL convention)."""
    src, y = supervised_positions(labels, ignore_index)
    h = hidden[src].float()
    r = torch.softmax(logits[src].float(), dim=-1)
    r[torch.arange(r.shape[0], device=r.device), y] -= 1.0
    return h, r


def output_layer_gradient(h: torch.Tensor, r: torch.Tensor, weights: torch.Tensor | None = None,
                          mean: bool = False) -> torch.Tensor:
    """Exact g = sum_t w_t r_t h_t^T of shape (V, d); ``mean=True`` divides by the token count
    (the mean-reduced loss used by GRACE)."""
    if h.shape[0] != r.shape[0]:
        raise ValueError("h and r must have the same number of tokens")
    g = (r if weights is None else r * weights.unsqueeze(-1)).T @ h
    if mean:
        if h.shape[0] == 0:
            raise ValueError("mean reduction over zero supervised tokens")
        g = g / h.shape[0]
    return g
