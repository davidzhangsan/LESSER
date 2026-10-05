"""Bit-exact equivalence of the ``lesser`` core with the implementations behind the paper.

Run: python -m unittest discover -s tests -v
The last test recomputes a released SFT selection from pool and query features; it is skipped unless
LESSER_ARTIFACTS points to a directory holding sft/features/llama-3.2-3b/ (written by `python -m sft.extract`).
"""
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lesser import RL_PAPER, SFT_PAPER, TwoSidedProjection, output_layer_gradient, round_robin, token_factors  # noqa: E402
from lesser.checks import check_exactness  # noqa: E402
from lesser.output_grad import token_factors_from_logits  # noqa: E402
from reference import original_impls as ref  # noqa: E402

V, D, T = 1000, 96, 40


def _example(seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(T, D, generator=g)
    W = torch.randn(V, D, generator=g) / D ** 0.5
    labels = torch.randint(0, V, (T,), generator=g)
    labels[: T // 3] = -100
    return hidden, W, labels


class TestCore(unittest.TestCase):
    def test_sft_projection_matches_extraction(self):
        hidden, W, labels = _example(1)
        h, r = token_factors(hidden, W, labels)
        h_ref, r_ref = ref.sft_forward_factors(hidden, W, labels)
        self.assertTrue(torch.equal(h, h_ref) and torch.equal(r, r_ref))
        ours = TwoSidedProjection(V, D, SFT_PAPER)
        theirs = ref.SFTProjectors(V, D, "cpu")
        self.assertTrue(torch.equal(ours.P_h, theirs.P_left) and torch.equal(ours.P_v, theirs.P_right))
        self.assertTrue(torch.equal(ours.project(h, r), theirs.prod(h_ref, r_ref)))

    def test_rl_projection_matches_scorer(self):
        hidden, W, labels = _example(2)
        src = (labels != -100).nonzero().flatten()
        h, lab = hidden[src], labels[src]
        logits = h @ W.T
        w = torch.rand(len(src))
        sk = ref.RLProdSketch(V, D, "cpu")
        feat = torch.zeros(64, 128)
        ref.rl_accumulate(feat, h, logits, lab, w, sk, exact=False)
        ours = TwoSidedProjection(V, D, RL_PAPER)
        self.assertTrue(torch.equal(ours.P_v, sk.R_v) and torch.equal(ours.P_h, sk.R_h))
        probs = torch.softmax(logits.float(), dim=-1)
        self.assertTrue(torch.equal(ours.project_from_probs(h, probs, lab, w), feat.reshape(-1)))
        exact = torch.zeros(V, D)
        ref.rl_accumulate(exact, h, logits, lab, w, None, exact=True)
        r = probs.clone(); r[torch.arange(len(lab)), lab] -= 1.0
        self.assertTrue(torch.allclose(output_layer_gradient(h, r, w), exact, atol=1e-6))

    def test_grace_mean_gradient(self):
        hidden, W, labels = _example(3)
        logits = hidden @ W.T
        # GRACE shifts labels itself: logits[t] predict labels[t+1]; token_factors uses the same convention
        h, r = token_factors_from_logits(hidden, logits, labels)
        G = output_layer_gradient(h, r, mean=True)
        self.assertTrue(torch.allclose(G, ref.grace_G(logits[None], hidden[None], labels[None]), atol=1e-6))

    def test_exactness_against_autograd(self):
        hidden, W, labels = _example(4)
        self.assertLess(check_exactness(hidden, W, labels), 1e-5)

    def test_round_robin_matches_nayak(self):
        rng = np.random.default_rng(0)
        for q in (1, 3, 9):
            s = rng.standard_normal((q, 500)).astype(np.float32)
            s[:, 7] = s[:, 3]  # ties break toward the lower index in both
            self.assertEqual(round_robin(s, 200), ref.nayak_round_robin(s, 200))

    @unittest.skipUnless(os.environ.get("LESSER_ARTIFACTS"), "set LESSER_ARTIFACTS to run")
    def test_released_selection_llama32_tydiqa(self):
        from lesser import select
        root = Path(os.environ["LESSER_ARTIFACTS"])
        pool = torch.load(root / "sft/features/llama-3.2-3b/pool_prod_0_197196.pt", map_location="cpu", weights_only=True)
        val = torch.load(root / "sft/features/llama-3.2-3b/val_tydiqa_prod.pt", map_location="cpu", weights_only=True)
        repo = Path(__file__).resolve().parents[1]
        want = json.loads((repo / "data/sft/selections/llama-3.2-3b/lesser/tydiqa_k1000.json").read_text())
        self.assertEqual(select(pool, val, budgets=(1000,))[1000], want)


if __name__ == "__main__":
    unittest.main()
