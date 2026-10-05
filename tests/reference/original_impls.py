"""Frozen copies of the implementations behind the paper's numbers, used only to test that the
``lesser`` package reproduces them bit for bit. Each block is copied from the file named above it,
with only imports and constants inlined.
"""
import math
from itertools import cycle

import numpy as np
import torch

# --- SFT: the extraction code behind the paper's SFT features (projectors and per-example factors) ---
PROJ_DIM, K_LEFT, K_RIGHT, TRAK_SEED = 8192, 128, 64, 0


class SFTProjectors:
    def __init__(self, vocab, d_hidden, device):
        gl = torch.Generator(device="cpu").manual_seed(TRAK_SEED)
        self.P_left = (torch.randint(0, 2, (d_hidden, K_LEFT), generator=gl, dtype=torch.int8)
                       .float().mul_(2).sub_(1)).to(device) / (K_LEFT ** 0.5)
        gr = torch.Generator(device="cpu").manual_seed(TRAK_SEED + 1)
        self.P_right = (torch.randint(0, 2, (vocab, K_RIGHT), generator=gr, dtype=torch.int8)
                        .float().mul_(2).sub_(1)).to(device) / (K_RIGHT ** 0.5)

    def prod(self, h_resp, r_resp):
        A = h_resp @ self.P_left
        B = r_resp @ self.P_right
        G = A.T @ B
        return G.reshape(-1)


@torch.no_grad()
def sft_forward_factors(h_all, W, labels):
    """forward_item after the model call: h_all is hidden_states[-1][0].float()."""
    W = W.float()
    sup_pos = (labels != -100).nonzero(as_tuple=False).flatten()
    sup_pos = sup_pos[sup_pos >= 1]
    src_pos = sup_pos - 1
    y = labels[sup_pos]
    h_resp = h_all[src_pos]
    p = torch.softmax(h_resp @ W.T, dim=-1)
    r = p.clone()
    r[torch.arange(r.shape[0]), y] -= 1.0
    return h_resp, r


# --- RL: GradAlign/select/parallel/prod_feature.py (sha 4a88c292, the version the RL runs used) -------
class RLProdSketch:
    def __init__(self, vocab_size, hidden_size, device, d_vocab=64, d_hidden=128, seed=0):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        rv_bits = torch.randint(0, 2, (vocab_size, d_vocab), generator=gen, dtype=torch.int8)
        rh_bits = torch.randint(0, 2, (hidden_size, d_hidden), generator=gen, dtype=torch.int8)
        self.R_v = (((rv_bits.to(torch.float32) * 2.0) - 1.0) / math.sqrt(d_vocab)).to(device)
        self.R_h = (((rh_bits.to(torch.float32) * 2.0) - 1.0) / math.sqrt(d_hidden)).to(device)


def rl_accumulate(feat, hidden, logits, labels, weights, sketch, exact):
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    if exact:
        probs[torch.arange(len(labels)), labels] -= 1.0
        feat += (probs * weights.unsqueeze(-1)).T @ hidden.to(torch.float32)
    else:
        r_proj = probs @ sketch.R_v - sketch.R_v[labels]
        h_proj = hidden.to(torch.float32) @ sketch.R_h
        feat += (r_proj * weights.unsqueeze(-1)).T @ h_proj


# --- GRACE: GRACE/GRACE/gradient_computation.py::compute_prod_features (exact G / n_sup) ---------------
def grace_G(logits, hidden, labels, ignore_index=-100):
    shift_logits, shift_hidden, shift_labels = logits[0, :-1], hidden[0, :-1], labels[0, 1:]
    sup_idx = (shift_labels != ignore_index).nonzero(as_tuple=True)[0]
    r = torch.softmax(shift_logits[sup_idx].float(), dim=-1)
    r[torch.arange(sup_idx.numel()), shift_labels[sup_idx]] -= 1.0
    G = r.t() @ shift_hidden[sup_idx].float()
    return G / int(sup_idx.numel())


# --- Selection: targeted-instruction-selection/selection/round_robin.py (Apache-2.0) ------------------
def nayak_round_robin(sim_matrix: np.ndarray, num_samples: int) -> list[int]:
    scores = sim_matrix.copy()
    used = np.zeros(scores.shape[1], bool)
    picked = []
    for i in cycle(range(scores.shape[0])):
        if len(picked) >= num_samples:
            break
        row = np.where(~used, scores[i], -np.inf)
        j = int(np.argmax(row))
        used[j] = True
        picked.append(j)
    return picked
