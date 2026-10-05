"""Cosine scoring and the round-robin selection rule of LESS / Nayak et al. (2026).

Round-robin cycles through the queries; at each step the current query takes its highest-scoring
pool item that has not been picked yet. This is the rule used for LESSER, LESS, and RDS+ in the
paper's SFT experiments; the code is equivalent to ``selection/round_robin.py`` in
https://github.com/dcml-lab/targeted-instruction-selection (Apache-2.0).
"""
from __future__ import annotations

from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """Float32, NaN/Inf-safe, unit-norm rows (zero rows stay zero)."""
    x = torch.nan_to_num(x.float().reshape(x.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, p=2, dim=1, eps=1e-12)


def cosine_scores(pool: torch.Tensor, queries: torch.Tensor, device: str = "cpu",
                  chunk: int = 16384) -> np.ndarray:
    """(num_queries, pool_size) cosine similarities in float32, computed in pool chunks."""
    q = normalize_rows(queries).to(device)
    p = normalize_rows(pool)
    out = np.empty((q.shape[0], p.shape[0]), dtype=np.float32)
    for s in range(0, p.shape[0], chunk):
        out[:, s:s + chunk] = (q @ p[s:s + chunk].to(device).T).cpu().numpy()
    return out


def round_robin(scores: np.ndarray, k: int) -> list[int]:
    """Greedy round-robin over queries (rows); ties break toward the lowest pool index."""
    if k > scores.shape[1]:
        raise ValueError("k exceeds the pool size")
    used = np.zeros(scores.shape[1], dtype=bool)
    picked: list[int] = []
    for i in cycle(range(scores.shape[0])):
        if len(picked) >= k:
            break
        j = int(np.argmax(np.where(~used, scores[i], -np.inf)))
        used[j] = True
        picked.append(j)
    return picked


def select(pool: torch.Tensor, queries: torch.Tensor, budgets=(1000, 5000, 10000),
           device: str = "cpu") -> dict[int, list[int]]:
    """Rank once to the largest budget and return nested prefixes for every budget."""
    order = round_robin(cosine_scores(pool, queries, device=device), max(budgets))
    return {k: order[:k] for k in budgets}
