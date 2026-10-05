"""Tests of the SFT component (CPU, no downloads, a few seconds apart from the extraction self-test).

Run: python -m unittest sft.test_sft -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sft import bins, extract, results, select
from sft.common import BUDGETS, DATA_DIR, MODEL_ORDER, N_POOL, TASKS, read_selection, selection_path
from sft.train_eval import train_command


def reference_full_order(sim: np.ndarray) -> list[int]:
    """The paper's bin builder: mask each picked column for every query."""
    s = sim.astype(np.float32, copy=True)
    picked, i = [], 0
    for _ in range(s.shape[1]):
        j = int(np.argmax(s[i]))
        picked.append(j)
        s[:, j] = -np.inf
        i = (i + 1) % s.shape[0]
    return picked


class TestExtraction(unittest.TestCase):
    def test_self_test(self):
        extract.self_test(verbose=False)


class TestSelection(unittest.TestCase):
    def test_pool_shards_must_tile(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("pool_prod_0_98598.pt", "pool_prod_98598_197196.pt"):
                (d / name).touch()
            self.assertEqual([p.name for p in select.pool_shards(d)],
                             ["pool_prod_0_98598.pt", "pool_prod_98598_197196.pt"])
            (d / f"pool_prod_0_{N_POOL}.pt").touch()      # a merged file next to its parts
            with self.assertRaises(ValueError):
                select.pool_shards(d)
            (d / "pool_prod_0_98598.pt").unlink()
            (d / f"pool_prod_0_{N_POOL}.pt").unlink()
            with self.assertRaises(ValueError):             # a gap
                select.pool_shards(d)

    def test_released_selections_are_nested(self):
        for model in MODEL_ORDER:
            for method in ("lesser", "less", "rds"):
                for task in TASKS:
                    lists = {k: read_selection(selection_path(model, method, task, k), k) for k in BUDGETS}
                    for k in BUDGETS[:-1]:
                        self.assertEqual(lists[k], lists[BUDGETS[-1]][:k], (model, method, task, k))


class TestBins(unittest.TestCase):
    def test_order_and_split_match_the_paper_builder(self):
        g = torch.Generator().manual_seed(0)
        pool, queries = torch.randn(3000, 16, generator=g), torch.randn(5, 16, generator=g)
        pool[7] = pool[3]                                   # a tie
        order = bins.pool_order(pool, queries)
        sim = bins._normalize_rows(queries) @ bins._normalize_rows(pool).T
        self.assertEqual(order, reference_full_order(sim))
        parts = bins.split_bins(order, n_bins=10, per_bin=50)
        self.assertEqual([len(p) for p in parts], [50] * 10)
        self.assertEqual(parts[9][0], order[9 * 300])


class TestReleasedResults(unittest.TestCase):
    def test_grid_cells(self):
        cells = results.downstream_cells(DATA_DIR)
        self.assertEqual(len(cells), 4 * 5 * 3 * 4)
        self.assertTrue(all(c["seeds"] == [0, 1, 2] for c in cells.values()))
        c = cells["llama-2-7b", "tydiqa", 1000, "lesser"]   # Table 9: 50.2 +- 0.4
        self.assertEqual((f"{c['mean']:.1f}", f"{c['std']:.1f}"), ("50.2", "0.4"))

    def test_bin_curves(self):
        curves = results.bin_curves(DATA_DIR)
        self.assertEqual(len(curves), 4 * 5 * 3)
        self.assertTrue(all(len(v) == 10 for v in curves.values()))


class TestTrainCommand(unittest.TestCase):
    def test_recipe(self):
        cmd = train_command("python", "meta-llama/Llama-2-7b-hf", Path("m"), Path("d.jsonl"), 1000, 2, "r", True)
        args = {tok: cmd[i + 1] for i, tok in enumerate(cmd[:-1])
                if tok.startswith("--") and not cmd[i + 1].startswith("--")}
        self.assertEqual(cmd[1:4], ["-u", "-m", "training.train_sft"])
        for key, value in {"--per_device_train_batch_size": "1", "--gradient_accumulation_steps": "128",
                           "--num_train_epochs": "2", "--learning_rate": "2e-5", "--seed": "2",
                           "--warmup_ratio": "0.03", "--lr_scheduler_type": "linear", "--weight_decay": "0.0",
                           "--num_samples": "1000"}.items():
            self.assertEqual(args.get(key), value, key)
        self.assertIn("--bf16", cmd)
        self.assertEqual(cmd[-2:], ["--gradient_checkpointing", "true"])


if __name__ == "__main__":
    unittest.main()
