"""Five-shot MMLU-Pro evaluation, as used for every MMLU-Pro number in the paper (Tables 9--11, Figure 3).

Runs lm-eval's ``mmlu_pro`` task (all 12,032 test questions) through Nayak et al.'s lm-eval wrapper settings: vLLM
backend, the Tulu chat template, batch size "auto", with ``num_fewshot=5`` (the task default) instead of the wrapper's
zero-shot. Writes ``<out_dir>/<tag>_extract.json`` with the accuracy (``exact_match,custom-extract``) and how often the
generation contains an extractable answer, plus per-question ``<tag>_samples.jsonl``.

Must run from the root of the Nayak et al. repository (it imports ``evaluation.lm_eval``) and as a module:
  cd third_party/targeted-instruction-selection/src
  PYTHONPATH=<release root>:. python -m sft.mmlu_pro_5shot --model_dir MODEL --out_dir DIR --tag NAME
(``sft/train_eval.py`` does this). This is the evaluation behind the paper's MMLU-Pro numbers; optional VLLM_*
environment variables are passed to vLLM as extra model arguments (the paper's runs set none).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REGEX = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")   # the task's answer filter (lm_eval tasks/mmlu_pro)
VLLM_ENV = {"VLLM_DTYPE": "dtype", "VLLM_GPU_MEM_UTIL": "gpu_memory_utilization", "VLLM_MAX_MODEL_LEN": "max_model_len",
            "VLLM_MAX_NUM_SEQS": "max_num_seqs", "VLLM_ENFORCE_EAGER": "enforce_eager"}


def vllm_model_args(base: str) -> str:
    extras = [f"{arg}={os.environ[name]}" for name, arg in VLLM_ENV.items() if os.environ.get(name)]
    return ",".join([base, *extras])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_fewshot", type=int, default=5)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    try:
        from evaluation.lm_eval import prepare_tokenizer_with_tulu_template   # Nayak et al.'s wrapper
    except ImportError as e:
        raise SystemExit(f"run from the Nayak et al. repository root ({e})")
    from lm_eval import evaluator

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok_dir = prepare_tokenizer_with_tulu_template(args.model_dir)
    results = evaluator.simple_evaluate(
        model="vllm", model_args=vllm_model_args(f"pretrained={args.model_dir},tokenizer={tok_dir}"),
        tasks=["mmlu_pro"], num_fewshot=args.num_fewshot, batch_size="auto", apply_chat_template=True,
        log_samples=True)
    res = results["results"]["mmlu_pro"]
    acc = float(res[next(k for k in res if k.startswith("exact_match") and "custom" in k)])
    # mmlu_pro is a task group: samples are keyed by subtask (mmlu_pro_biology, ...)
    samples = [s for k, v in results["samples"].items() if k.startswith("mmlu_pro") for s in v]
    n = len(samples)
    n_ext = n_correct = n_ext_correct = 0
    openings, rows = {}, []
    for s in samples:
        gen = s["resps"][0][0] if s.get("resps") else ""
        em = float(s.get("exact_match", 0.0))
        ext = REGEX.search(gen) is not None
        n_ext += ext
        n_correct += em > 0
        n_ext_correct += (em > 0) and ext
        head = gen.strip()[:12]
        openings[head] = openings.get(head, 0) + 1
        rows.append(dict(ext=ext, em=em, gen_len=len(gen)))
    summary = dict(tag=args.tag, model_dir=args.model_dir, num_fewshot=args.num_fewshot, n=n, accuracy=acc,
                   accuracy_from_samples=n_correct / n, extractable_fraction=n_ext / n,
                   accuracy_given_extractable=(n_ext_correct / n_ext) if n_ext else None,
                   correct_but_not_extractable=(n_correct - n_ext_correct) / n,
                   median_generation_chars=sorted(r["gen_len"] for r in rows)[n // 2],
                   top_generation_openings=sorted(openings.items(), key=lambda kv: -kv[1])[:8],
                   seconds=round(time.time() - t0))
    (out / f"{args.tag}_extract.json").write_text(json.dumps(summary))
    (out / f"{args.tag}_samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"EXTRACT {args.tag}: accuracy {100 * acc:.2f} | extractable {100 * summary['extractable_fraction']:.1f}%",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
