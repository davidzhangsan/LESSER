# targeted-instruction-selection (Nayak et al., 2026)

The SFT experiments train and evaluate with the released code of Nayak et al., *A critical look at targeted
instruction selection*, and compare against their released results and selections.

- Upstream: https://github.com/dcml-lab/targeted-instruction-selection
- Pinned commit: `8ff397ad1c6e649465948a7ce847cb003140224c` (2026-02-25)
- License: Apache License 2.0; a copy is in `LICENSE`. `sft/nayak_data.py` copies two functions of `common/data.py`,
  and `data/sft/nayak/` holds unmodified copies of the repository's result tables (`assets/plot_data/quantile_budget/`).
- Nayak et al.'s LESS and RDS+ selections (`data/sft/selections/*/{less,rds}/`) come from their Hugging Face datasets
  `Harvard-DCML/tis-subset-datasets-*`, not from this repository; see `data/sft/README.md` for their license status.

## Setup

```bash
cd third_party/targeted-instruction-selection
git clone https://github.com/dcml-lab/targeted-instruction-selection src
cd src
git checkout 8ff397ad1c6e649465948a7ce847cb003140224c
git apply ../lesser_sft.patch
bash download_eval.sh                    # evaluation data -> data/eval
pip install -r requirements.txt vllm lm-eval
```

`sft/train_eval.py` expects this checkout at `third_party/targeted-instruction-selection/src` (or `--nayak-root`).
OLMo3-7B needs transformers >= 4.57 and a vLLM version that supports it (the paper's OLMo3 runs used vLLM 0.11.0).

## What `lesser_sft.patch` changes

Both changes are switched by environment variables. With the variables unset the code is the upstream code.
`sft/train_eval.py` sets them as the paper's runs had them.

| File | Variable | Values | Effect |
|---|---|---|---|
| `training/train_sft.py` | `TIS_PAD_TOKEN_MODE` | `always` (default, upstream), `if_missing` | Upstream always adds a `[PAD]` token and resizes the embeddings. `if_missing` keeps a tokenizer's own pad token. Llama-2-7B and Llama-3.2-3B have no pad token, so both settings are identical for them; Qwen3-4B-Base and OLMo3-7B have one. Training uses per-device batch 1, so the pad token never appears in a batch; the resize changes the vocabulary size of the fine-tuned model. |
| `evaluation/gsm/run_eval.py` | `TIS_GSM_CHAT_STOP` | `none` (default, upstream), `blank_line` | `blank_line` stops chat-format generations at `"\n\n"`; upstream sets no stop sequence in chat format. |

## Which code produced which numbers

The paper's runs used two kinds of machines, a training workstation and a GPU cluster (the `env` field of the released
cell records). "Verified" means the file checksums of the code used were compared with the pinned commit and with
the patched tree; "not verified" means no such comparison was possible.

| Paper numbers | Code | `TIS_PAD_TOKEN_MODE` | `TIS_GSM_CHAT_STOP` |
|---|---|---|---|
| Random, RDS+, LESS on TyDiQA, GSM8K, Codex, BBH (Tables 9--11, Figure 3) | Nayak et al.'s own runs (released CSVs) | upstream | upstream |
| LESSER on TyDiQA, GSM8K, Codex, BBH for Llama-3.2-3B, Llama-2-7B, Qwen3-4B-Base; the Qwen3 LESS rerun of Appendix A.1 | upstream code on the workstation; not verified | `always` if unmodified | `none` if unmodified |
| LESSER on TyDiQA, GSM8K, Codex, BBH for OLMo3-7B; the OLMo3 LESS rerun of Appendix A.1 | a later upstream commit on the cluster (verified): evaluation files identical to 8ff397a; `train_sft.py` differs only by an option for another dataset that is inactive here | `always` | `none` |
| MMLU-Pro, all four methods and models (Tables 9--11, Figure 3) | the patched tree on the cluster (verified); evaluation by `sft/mmlu_pro_5shot.py` | `if_missing` | not used |
| Llama-3.2-3B LESS and Random reruns of Appendix A.1 | the patched tree on the workstation; not verified | same result either way (no pad token) | `blank_line` if the code matched the patched tree |
| Similarity-bin cells (Figures 4, 8--10) | Llama-3.2-3B: the patched `train_sft.py` (verified); Llama-2-7B and Qwen3-4B-Base: the patched tree on the workstation, not verified (Qwen3 scored with the repository's `evaluation/ce_loss.py`); OLMo3-7B and the Llama-2 Codex rerun: the later upstream commit on the cluster (verified) | Qwen3: `if_missing` if the code matched the patched tree; OLMo3: `always` | not used |

The code used for the runs also contained changes to other files (vLLM memory options, LESS extraction memory, progress
logging). None of these changes affects a reported SFT number, so they are not part of the patch.

## Upstream issues that `sft/train_eval.py` works around

- `evaluation/run_eval.py` sends `mmlu_pro` to `evaluation.custom_eval`, which is not in the repository. The paper
  evaluates MMLU-Pro five-shot with `sft/mmlu_pro_5shot.py`, which calls lm-eval with the repository's wrapper settings.
- `evaluation/run_eval.py` starts `python3 -m evaluation.<task>.run_eval`; `sft/train_eval.py` puts the evaluation
  interpreter first on `PATH`.
- `selection/round_robin.py` uses `tqdm` without importing it; the release uses `lesser.round_robin`, which the core
  tests check against a copy of this function.
- `training/train_sft.py` can leave a child process holding GPU memory after training; `sft/train_eval.py` runs each
  step in its own process group and ends the group when the step exits.
