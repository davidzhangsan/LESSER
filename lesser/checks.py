"""Exactness check: the forward-only output-layer gradient equals autograd on a readout copy.

Proposition 1 differentiates with respect to a copy W_out of the final projection that is used only
as the readout, so h_t does not depend on it. This check recomputes that gradient with autograd for
one example (sum-reduced cross-entropy over supervised tokens) and compares it with
``output_layer_gradient``.
"""
from __future__ import annotations

import torch

from .output_grad import output_layer_gradient, supervised_positions, token_factors


def autograd_readout_gradient(hidden: torch.Tensor, W: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """d/dW_out of sum_t -log softmax(W_out h_t)[y_t], with hidden states treated as constants."""
    src, y = supervised_positions(labels)
    W_out = W.detach().float().clone().requires_grad_(True)
    logits = hidden[src].detach().float() @ W_out.T
    loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, W_out)
    return grad


def check_exactness(hidden: torch.Tensor, W: torch.Tensor, labels: torch.Tensor, rtol: float = 1e-4) -> float:
    """Return the relative Frobenius error between the forward-only and autograd gradients; raise if above rtol."""
    h, r = token_factors(hidden, W, labels)
    ours = output_layer_gradient(h, r)
    ref = autograd_readout_gradient(hidden, W, labels)
    err = float((ours - ref).norm() / ref.norm().clamp_min(1e-30))
    if err > rtol:
        raise AssertionError(f"output-layer gradient differs from autograd: relative error {err:.2e}")
    return err
