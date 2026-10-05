"""Recipe SFT run that records the query loss at every step, for the training trajectories of Appendix F.3.

Training is ``training/train_sft.py`` of targeted-instruction-selection (Nayak et al., 2026) as used for
the paper's SFT runs: full fine-tuning of the bf16 model with HF Trainer on the selected examples,
encoded by ``sft.nayak_data.encode_with_messages_format`` (max length 2048). One local difference from the
released script, shared with the paper's SFT runs: a "[PAD]" token is added (and the embeddings resized)
only when the tokenizer has no pad token. The released script's LoRA branch is not included, since every
run here is full fine-tuning.

``--dev_task T --loss_out F`` adds a TrainerCallback that records the mean per-query response-token loss on
the T dev queries before the first update and after every optimizer step (bf16 model, no_grad, one query
per forward). It does not change the optimization. Used for Figure 7b and the per-pair query-loss figure of
Appendix F.3. When ``--loss_out`` is set, no weights are saved.

Run through ``python -m analysis.trajectories train`` (recipe flags) or directly from the repository root:
  python -m analysis.train_sft_probe --model_name SNAPSHOT --output_dir RUN/model \\
      --per_device_train_batch_size 1 --gradient_accumulation_steps 128 --num_train_epochs 2 \\
      --learning_rate 2e-5 --seed 0 --warmup_ratio 0.03 --lr_scheduler_type linear --weight_decay 0.0 \\
      --logging_steps 1 --bf16 --train_dataset_path RUN/train.jsonl --num_samples 5000 --report_to none \\
      --save_strategy no --dev_task gsm8k --loss_out RUN/loss_trajectory.json
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Run by path: import the analysis and sft packages from the repository root, not from this directory.
    _here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _here]
    sys.path.insert(0, str(_here.parent))

import torch
from transformers import TrainerCallback

from analysis import common


@dataclass
class TrainingConfig:
    train_dataset_name: Optional[str] = field(default=None, metadata={"help": "training dataset on the hub"})
    train_dataset_config_name: Optional[str] = field(default=None, metadata={"help": "config of the hub dataset"})
    train_dataset_path: str = field(default=None, metadata={"help": "local JSON-lines training file with 'messages'"})
    num_samples: Optional[int] = field(default=None, metadata={"help": "use the first num_samples rows"})
    model_name: Optional[str] = field(default="meta-llama/Llama-3.2-3B", metadata={"help": "base model"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "use Flash Attention 2"})
    dev_task: Optional[str] = field(default=None, metadata={"help": "task whose dev queries are scored after every step"})
    loss_out: Optional[str] = field(default=None, metadata={"help": "output JSON of the query-loss trajectory"})


class DevLossCallback(TrainerCallback):
    """Mean per-query response-token loss on the dev queries at step 0 and after every optimizer step."""

    def __init__(self, dev, out_path, tag):
        self.dev, self.out, self.tag, self.rows = dev, out_path, tag, []

    def _write(self, complete):
        with open(self.out, "w") as f:
            json.dump({"tag": self.tag, "n_queries": len(self.rows[0]["per_query"]), "complete": complete,
                       "rows": self.rows}, f)

    def _measure(self, model, step):
        was_training = model.training
        model.eval()
        losses = []
        with torch.no_grad():
            for q in self.dev:
                b = {k: torch.as_tensor(q[k], dtype=torch.long)[None].to(model.device)
                     for k in ("input_ids", "attention_mask", "labels")}
                losses.append(float(model(**b).loss))
        if was_training:
            model.train()
        self.rows.append({"step": int(step), "mean_query_loss": sum(losses) / len(losses), "per_query": losses})
        self._write(False)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._measure(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._measure(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._write(True)


def train():
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                              TrainingArguments, set_seed)
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_info()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger = hf_logging.get_logger(__name__)
    parser = HfArgumentParser((TrainingArguments, TrainingConfig))
    hf_args, cfg = parser.parse_args_into_dataclasses()
    from sft.nayak_data import construct_test_sample, encode_with_messages_format
    set_seed(hf_args.seed)

    kwargs = {"attn_implementation": "flash_attention_2"} if cfg.use_flash_attention_2 else {}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    if cfg.train_dataset_name is not None:
        logger.info(f"Loading dataset from hub: {cfg.train_dataset_name}")
        train_dataset = load_dataset(cfg.train_dataset_name, cfg.train_dataset_config_name)["train"]
    else:
        logger.info(f"Loading dataset from local path: {cfg.train_dataset_path}")
        train_dataset = load_dataset("json", data_files=cfg.train_dataset_path)["train"]
    if cfg.num_samples is not None:
        train_dataset = train_dataset.select(list(range(cfg.num_samples)))
    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(x, tokenizer, 2048))
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # drop examples whose labels are all -100 (longer than the maximum length)
    train_dataset = train_dataset.filter(lambda x: (x["labels"] != -100).any())
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    callbacks = []
    if cfg.dev_task is not None:
        if cfg.loss_out is None:
            raise RuntimeError("--dev_task needs --loss_out")
        dev = load_dataset(common.QUERY_DATASET, cfg.dev_task, split="dev")
        dev = dev.map(lambda x: construct_test_sample(sample=x, tokenizer=tokenizer, max_length=2048))
        callbacks.append(DevLossCallback(dev, cfg.loss_out, f"{cfg.dev_task}:{hf_args.run_name}"))

    trainer = Trainer(model=model, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=collator, args=hf_args,
                      callbacks=callbacks)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.train()
    if cfg.loss_out is None:
        trainer.save_model(hf_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    trainer.save_state()


if __name__ == "__main__":
    train()
