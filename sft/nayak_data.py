"""Tokenization and loss masking of Nayak et al. (2026), used for LESSER pool and query features.

Copied from ``common/data.py`` of https://github.com/dcml-lab/targeted-instruction-selection at commit 8ff397a
(Apache License 2.0; copyright the original authors), keeping only the default code path that training
(``training/train_sft.py``), LESS extraction and the query loss use. Behavior is unchanged: pool examples are the
Tulu chat format with only assistant tokens (and their end-of-sequence token) supervised; query examples are the
prompt followed by the label, with only the label supervised.
"""
from __future__ import annotations

import torch


def encode_with_messages_format(example, tokenizer, max_seq_length=2048):
    """Tokenize a ``messages`` example; labels are -100 everywhere except assistant content."""
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors="pt",
                    max_length=max_seq_length, truncation=True).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(messages_so_far, return_tensors="pt", max_length=max_seq_length,
                                        truncation=True).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids.flatten(), "labels": labels.flatten(), "attention_mask": attention_mask.flatten()}


def construct_test_sample(tokenizer, sample, max_length=2048):
    """Tokenize a query (``prompts``, ``labels``); prompt and label are tokenized separately, then concatenated."""
    prompt, label = sample["prompts"], sample["labels"]
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    labels = tokenizer(label + tokenizer.eos_token, return_tensors="pt", max_length=max_length, truncation=True,
                       add_special_tokens=False)
    return {
        "input_ids": torch.cat([inputs["input_ids"][0], labels["input_ids"][0]], dim=0),
        "attention_mask": torch.cat([inputs["attention_mask"][0], labels["attention_mask"][0]], dim=0),
        "labels": torch.cat([torch.ones_like(inputs["input_ids"][0]) * -100, labels["input_ids"][0]], dim=0),
    }
