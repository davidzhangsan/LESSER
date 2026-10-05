"""One downstream SFT cell: a selection -> full fine-tuning -> evaluation -> one result record (GPU).

Paper results: the LESSER cells of Tables 9--11 and Figure 3 (TyDiQA, GSM8K, Codex, BBH), the five-shot MMLU-Pro
reruns of all four methods (the MMLU-Pro rows of Tables 9--11 and Figure 3), and the replication checks of
Appendix A.1. The other baseline cells are Nayak et al.'s released scores. ``sft/bins.py cell`` reuses this runner
for the similarity-bin cells.

Training calls Nayak et al.'s ``training.train_sft`` (third_party/targeted-instruction-selection, pinned at 8ff397a
plus ``lesser_sft.patch``) with the paper's recipe: full fine-tuning, per-device batch 1 x gradient accumulation 128,
2 epochs, learning rate 2e-5, linear schedule with 3% warmup, weight decay 0, bf16, maximum length 2,048, gradient
checkpointing for the 7B models, and the cell's seed. Evaluation:
  * TyDiQA (F1), GSM8K (exact match), Codex (pass@10), BBH (exact match): Nayak et al.'s ``evaluation.run_eval`` with
    vLLM; data from their ``download_eval.sh``.
  * MMLU-Pro: five-shot lm-eval ``mmlu_pro`` on all 12,032 test questions (``sft/mmlu_pro_5shot.py``).

Selections: ``--method lesser|less|rds|random`` reads ``data/sft/selections/<model>/<method>/<task>_k<k>.json``
(``random`` exists for MMLU-Pro only: the fixed subsets of the five-shot reruns); ``--method random-draw`` draws a
fresh subset with ``numpy.random.default_rng(seed).choice(197196, k, replace=False)`` (the Random reruns of
Appendix A.1); ``--selection FILE`` trains on an explicit list under the label given by ``--method``.

Two code-path switches of Nayak et al.'s repository, set as in the paper's runs (see third_party/.../README.md):
  --pad-token-mode  always (upstream: add a [PAD] token and resize) | if_missing (keep a tokenizer's own pad token).
                    Default: if_missing for MMLU-Pro (the five-shot reruns), always otherwise. Differs only for Qwen3
                    and OLMo3, whose tokenizers have pad tokens.
  --gsm-chat-stop   none (upstream) | blank_line (stop generation at "\\n\\n"; used by the Appendix A.1 reruns).

Output (``--out-dir``, default outputs/sft): results/<model>_<method>_<task>_k<k>_s<seed>.json with the keys of the
paper's cell records (model, method, task, budget, seed, metric_name, metric_value, n_selected, base_model, env) plus
the switches and selection used; an existing record is not recomputed.

Run:
  python -m sft.train_eval --model llama-2-7b --method lesser --task tydiqa --budget 1000 --seed 0 \\
      --nayak-root third_party/targeted-instruction-selection/src --train-python PY --eval-python PY
  add --dry-run to print the commands without running them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import (DATA_DIR, DEFAULT_OUT, MAX_SEQ_LENGTH, N_POOL, POOL_DATASET, QUERY_DATASET, REPO_ROOT, TASKS,
                     model_spec, read_json, read_selection, selection_path, write_json)

TASK_METRIC = {"tydiqa": "f1", "gsm8k": "exact_match", "codex": "pass@10", "bbh": "average_exact_match",
               "mmlu_pro": "exact_match,custom-extract (5-shot)"}
DEFAULT_NAYAK = REPO_ROOT / "third_party" / "targeted-instruction-selection" / "src"

# Dev-query cross-entropy of a model: mean over the task's dev queries of the per-query mean token CE under Nayak
# et al.'s construct_test_sample (the convention of every similarity-bin cell). Runs in the training interpreter.
DEV_CE_CODE = """
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from common.data import construct_test_sample
model_dir, task, out, dataset, max_length = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
dev = load_dataset(dataset, task, split="dev")
losses = []
with torch.no_grad():
    for j in range(len(dev)):
        e = construct_test_sample(sample=dev[j], tokenizer=tok, max_length=max_length)
        ids = torch.tensor(e["input_ids"]).unsqueeze(0).cuda(); lab = torch.tensor(e["labels"]).unsqueeze(0).cuda()
        losses.append(float(model(input_ids=ids, labels=lab).loss))
assert losses, "empty dev set"
json.dump(dict(dev_ce=sum(losses) / len(losses), n_dev=len(losses), per_query=losses,
               convention="mean over dev queries of construct_test_sample CE"), open(out, "w"))
"""


def cell_name(model: str, method: str, task: str, budget: int, seed: int) -> str:
    return f"{model}_{method}_{task}_k{budget}_s{seed}"


def resolve_selection(args) -> tuple[list[int], str]:
    if args.selection:
        return read_selection(args.selection, args.budget), str(args.selection)
    if args.method == "random-draw":
        idx = np.random.default_rng(args.seed).choice(N_POOL, args.budget, replace=False).tolist()
        how = f"numpy.random.default_rng({args.seed}).choice({N_POOL}, {args.budget}, replace=False)"
        return [int(i) for i in idx], how
    path = selection_path(args.model, args.method, args.task, args.budget, args.data_dir)
    return read_selection(path, args.budget), str(path)


def materialize(indices: list[int], out_jsonl: Path) -> None:
    """Write the selected pool rows as the {"messages": ...} JSONL that train_sft reads."""
    from datasets import load_dataset
    ds = load_dataset(POOL_DATASET, split="train")
    if len(ds) != N_POOL:
        raise RuntimeError(f"{POOL_DATASET}: {len(ds)} rows, expected {N_POOL}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_name(out_jsonl.name + ".tmp")
    with open(tmp, "w") as f:
        for ex in ds.select(indices):
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")
    tmp.replace(out_jsonl)


def train_command(train_python: str, hf_id: str, model_dir: Path, jsonl: Path, budget: int, seed: int, run_name: str,
                  gradient_checkpointing: bool) -> list[str]:
    cmd = [train_python, "-u", "-m", "training.train_sft", "--model_name", hf_id, "--output_dir", str(model_dir),
           "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "128", "--num_train_epochs", "2",
           "--learning_rate", "2e-5", "--seed", str(seed), "--warmup_ratio", "0.03", "--lr_scheduler_type", "linear",
           "--weight_decay", "0.0", "--save_strategy", "no", "--logging_steps", "5", "--bf16",
           "--train_dataset_path", str(jsonl), "--num_samples", str(budget), "--report_to", "none",
           "--run_name", run_name]
    return cmd + (["--gradient_checkpointing", "true"] if gradient_checkpointing else [])


def eval_command(eval_python: str, task: str, model_dir: Path, save_dir: Path, eval_data_dir: Path,
                 tag: str) -> list[str]:
    if task == "mmlu_pro":
        return [eval_python, "-u", "-m", "sft.mmlu_pro_5shot", "--model_dir", str(model_dir),
                "--out_dir", str(save_dir), "--tag", tag, "--num_fewshot", "5"]
    return [eval_python, "-u", "-m", "evaluation.run_eval", "--model_name_or_path", str(model_dir), "--eval_dataset",
            task, "--eval_data_dir", str(eval_data_dir), "--save_dir", str(save_dir), "--use_vllm"]


def run(cmd: list[str], cwd: Path, env: dict) -> None:
    """Run a command in its own process group and clean up any children it leaves behind (e.g. GPU workers)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    try:
        rc = proc.wait()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def read_metric(task: str, save_dir: Path, tag: str) -> tuple[str, float]:
    """The paper's metric on a 0--100 scale."""
    if task == "mmlu_pro":
        rec = read_json(save_dir / f"{tag}_extract.json")
        if rec["num_fewshot"] != 5 or rec["n"] != 12032:
            raise ValueError(f"{save_dir}: not a five-shot full-test evaluation")
        return TASK_METRIC[task], 100 * float(rec["accuracy"])
    found = list(save_dir.rglob("metrics.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one metrics.json under {save_dir}, found {len(found)}")
    d = json.loads(found[0].read_text())
    if task == "tydiqa":
        return "f1", float(d["average"]["f1"])
    key = TASK_METRIC[task]
    avg = d.get("average")
    raw = avg[key] if isinstance(avg, dict) and key in avg else d.get(key)
    if raw is None:
        raise KeyError(f"{found[0]}: no {key!r}")
    return key, 100.0 * float(raw)


def dev_query_ce(train_python: str, model_dir: Path, task: str, out_json: Path, cwd: Path, env: dict) -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    run([train_python, "-u", "-c", DEV_CE_CODE, str(model_dir), task, str(out_json), QUERY_DATASET,
         str(MAX_SEQ_LENGTH)], cwd, env)
    return read_json(out_json)


def run_cell(args) -> dict | None:
    spec = model_spec(args.model)
    name = cell_name(args.model, args.method, args.task, args.budget, args.seed)
    out = Path(args.out_dir)
    result_path = out / "results" / f"{name}.json"
    if result_path.exists():
        print(f"[cell] {name}: result exists, skipping", flush=True)
        return read_json(result_path)
    nayak = Path(args.nayak_root)
    if not (nayak / "training" / "train_sft.py").is_file():
        raise FileNotFoundError(f"{nayak}: not a targeted-instruction-selection checkout (see third_party/...)")
    pad_mode = args.pad_token_mode or ("if_missing" if args.task == "mmlu_pro" else "always")
    indices, selection = resolve_selection(args)
    jsonl, model_dir, save_dir = out / "train_data" / f"{name}.jsonl", out / "models" / name, out / "eval" / name
    tag = f"{name}_5shot"
    base_env = dict(os.environ, TOKENIZERS_PARALLELISM="false", TIS_PAD_TOKEN_MODE=pad_mode,
                    TIS_GSM_CHAT_STOP=args.gsm_chat_stop)
    if args.gpu is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_env = dict(base_env, PYTHONPATH=str(nayak))
    # run_eval.py starts `python3 -m evaluation.<task>.run_eval`: put the evaluation interpreter first on PATH
    eval_env = dict(base_env, PYTHONPATH=os.pathsep.join([str(nayak), str(REPO_ROOT)]),
                    PATH=os.pathsep.join([str(Path(args.eval_python).parent), os.environ.get("PATH", "")]))
    train_cmd = train_command(args.train_python, spec.hf_id, model_dir, jsonl, args.budget, args.seed, name,
                              spec.gradient_checkpointing)
    eval_data = Path(args.eval_data_dir or nayak / "data" / "eval")
    eval_cmd = eval_command(args.eval_python, args.task, model_dir, save_dir, eval_data, tag)
    if args.dry_run:
        print(json.dumps({"cell": name, "selection": selection, "n_selected": len(indices),
                          "first_indices": indices[:5],
                          "TIS_PAD_TOKEN_MODE": pad_mode, "TIS_GSM_CHAT_STOP": args.gsm_chat_stop,
                          "train": train_cmd, "eval": None if args.no_eval else eval_cmd, "cwd": str(nayak)}, indent=1))
        return None

    t0 = time.time()
    materialize(indices, jsonl)
    print(f"[cell] {name}: training on {len(indices)} examples", flush=True)
    run(train_cmd, nayak, train_env)
    record = dict(model=args.model, method=args.method, task=args.task, budget=args.budget, seed=args.seed,
                  base_model=spec.hf_id, n_selected=len(indices), selection=selection, env=args.env_tag,
                  pad_token_mode=pad_mode, gsm_chat_stop=args.gsm_chat_stop)
    if args.dev_ce:
        ce = dev_query_ce(args.train_python, model_dir, args.task, save_dir / "dev_ce.json", nayak, train_env)
        record.update(dev_ce=ce["dev_ce"], n_dev=ce["n_dev"], per_query=ce["per_query"], convention=ce["convention"])
    if not args.no_eval:
        print(f"[cell] {name}: evaluating", flush=True)
        run(eval_cmd, nayak, eval_env)
        record["metric_name"], record["metric_value"] = read_metric(args.task, save_dir, tag)
    record["seconds"] = round(time.time() - t0)
    write_json(record, result_path, indent=2)
    if not args.keep_model:
        shutil.rmtree(model_dir, ignore_errors=True)
    print(f"[cell] {name}: " + (f"{record['metric_name']} = {record['metric_value']:.2f}" if "metric_value" in record
                                 else f"dev CE = {record['dev_ce']:.4f}"), flush=True)
    return record


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--nayak-root", type=Path, default=DEFAULT_NAYAK,
                    help="checkout of targeted-instruction-selection @8ff397a with lesser_sft.patch applied")
    ap.add_argument("--train-python", default=sys.executable, help="interpreter with Nayak et al.'s training stack")
    ap.add_argument("--eval-python", default=sys.executable, help="interpreter with vLLM and lm-eval")
    ap.add_argument("--eval-data-dir", type=Path, help="default: <nayak-root>/data/eval (download_eval.sh)")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gpu", help="value for CUDA_VISIBLE_DEVICES")
    ap.add_argument("--pad-token-mode", choices=["always", "if_missing"],
                    help="default: if_missing for mmlu_pro, always otherwise (as in the paper)")
    ap.add_argument("--gsm-chat-stop", choices=["none", "blank_line"], default="none")
    ap.add_argument("--env-tag", default="release", help="free-form label stored as the record's env")
    ap.add_argument("--keep-model", action="store_true")
    ap.add_argument("--dry-run", action="store_true")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True,
                    help="lesser | less | rds | random | random-draw | a label for --selection")
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--selection", type=Path, help="explicit selection file (JSON list of pool indices)")
    ap.add_argument("--dev-ce", action="store_true", help="also record the trained model's dev-query cross-entropy")
    ap.add_argument("--no-eval", action="store_true", help="skip the downstream evaluation")
    add_arguments(ap)
    args = ap.parse_args(argv)
    if args.no_eval and not args.dev_ce:
        ap.error("--no-eval needs --dev-ce (nothing would be recorded)")
    if args.selection is None and args.method not in ("lesser", "less", "rds", "random", "random-draw"):
        ap.error("a free-form --method label needs --selection")
    run_cell(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
