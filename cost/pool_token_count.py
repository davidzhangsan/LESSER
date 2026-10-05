"""Exact token counts of the 197,196-example SFT candidate pool under the extraction format.

Paper: Appendix A.4, Table 5 (processed tokens T = 86,343,509 and response tokens T_r = 25,430,302
for the Llama-2 tokenizer). These counts feed cost/sft_flops.py.

The pool is Harvard-DCML/tulu-v2-197K-processed, train split. Each example is encoded with
``sft.nayak_data.encode_with_messages_format``, the LESS chat format of dcml-lab/targeted-instruction-
selection (common/data.py, commit 8ff397a) that the SFT features and training use: messages joined as
'<|user|>\\n...', '<|assistant|>\\n...' plus EOS, truncated to 2,048 tokens, labels on the assistant
tokens only. T counts the input ids after truncation, T_r the labels that are not -100, and
``truncated_at_2048`` the examples whose encoding reaches the 2,048-token cap.

Without --out the script compares its result with data/cost/pool_token_count.json, the
run behind the paper, and fails on any difference. Dataset and tokenizer revisions are pinned to
the ones that run used.

Needs ``datasets`` and ``transformers`` (``pip install -e .[sft]``), the dataset and the tokenizers
(the meta-llama repositories are gated). CPU only, about one minute per tokenizer with 16 processes.

    python -m cost.pool_token_count
    python -m cost.pool_token_count --tokenizers meta-llama/Llama-2-7b-hf --out counts.json
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from cost._common import DATA_DIR, POOL_EXAMPLES, require
from sft.common import MAX_SEQ_LENGTH, POOL_DATASET as DATASET
from sft.nayak_data import encode_with_messages_format

DATASET_REVISION = "e217e5f72f7a0d10748d4c61abc5856338d90c7f"
TOKENIZER_REVISIONS = {
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",   # the SFT FLOP model
    "meta-llama/Llama-3.2-3B": "13afe5124825b4f3751f836b40dafda64c1ed062",    # recorded for reference
}
IGNORE_INDEX = -100
RESULT = "pool_token_count.json"


_TOKENIZER = None


def _init_worker(name: str, revision: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)


def _count(messages: list[dict]) -> tuple[int, int]:
    """(tokens, response tokens) of one pool example."""
    enc = encode_with_messages_format({"messages": messages}, _TOKENIZER, MAX_SEQ_LENGTH)
    return int(enc["input_ids"].numel()), int((enc["labels"] != IGNORE_INDEX).sum())


def load_pool_messages() -> list[list[dict]]:
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", revision=DATASET_REVISION)
    if "messages" not in ds.column_names:
        raise ValueError(f"{DATASET}: no 'messages' column (columns: {ds.column_names})")
    if len(ds) != POOL_EXAMPLES:
        raise ValueError(f"{DATASET}: {len(ds)} examples, expected {POOL_EXAMPLES}")
    return list(ds["messages"])


def count_pool(messages: list[list[dict]], name: str, num_proc: int) -> dict:
    """Token statistics of the whole pool for one tokenizer, in the schema of pool_token_count.json."""
    if name not in TOKENIZER_REVISIONS:
        raise ValueError(f"no pinned revision for tokenizer {name!r}; known: {sorted(TOKENIZER_REVISIONS)}")
    with Pool(num_proc, initializer=_init_worker, initargs=(name, TOKENIZER_REVISIONS[name])) as pool:
        counts = pool.map(_count, messages, chunksize=256)
    tokens = sum(t for t, _ in counts)
    response = sum(r for _, r in counts)
    return dict(examples=len(counts), tokens=tokens, response_tokens=response, mean_tokens=tokens / len(counts),
                response_share=response / tokens, truncated_at_2048=sum(t >= MAX_SEQ_LENGTH for t, _ in counts))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Token counts of the SFT pool under the extraction format.")
    ap.add_argument("--tokenizers", nargs="+", default=list(TOKENIZER_REVISIONS))
    ap.add_argument("--num-proc", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None, help="write the counts here instead of comparing")
    args = ap.parse_args(argv)

    messages = load_pool_messages()
    result = {}
    for name in args.tokenizers:
        result[name] = count_pool(messages, name, args.num_proc)
        print(name, json.dumps(result[name]), flush=True)

    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")
        return
    recorded = json.loads(require(args.data_dir / RESULT).read_text())
    for name, stats in result.items():
        if recorded.get(name) != stats:
            raise SystemExit(f"{name}: recomputed {stats} differs from {RESULT}: {recorded.get(name)}")
    print(f"identical to {args.data_dir / RESULT} for {len(result)} tokenizer(s)")


if __name__ == "__main__":
    main()
