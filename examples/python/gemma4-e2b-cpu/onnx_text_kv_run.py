#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


PAST_KEY_RE = re.compile(r"^past_key_values\.(\d+)\.key$")
PAST_VALUE_RE = re.compile(r"^past_key_values\.(\d+)\.value$")
PRESENT_KEY_RE = re.compile(r"^present\.(\d+)\.key$")
PRESENT_VALUE_RE = re.compile(r"^present\.(\d+)\.value$")


def ort_type_to_numpy(ort_type: str):
    if ort_type == "tensor(float)":
        return np.float32
    if ort_type == "tensor(float16)":
        return np.float16
    if ort_type == "tensor(int64)":
        return np.int64
    if ort_type == "tensor(int32)":
        return np.int32
    if ort_type == "tensor(bool)":
        return np.bool_
    raise ValueError(f"Unsupported ORT tensor type: {ort_type}")


@dataclass
class KvInputSpec:
    key_name: str
    value_name: str
    present_key_name: str
    present_value_name: str
    kv_dtype: np.dtype
    num_kv_heads: int
    head_size: int


def discover_kv_specs(session: ort.InferenceSession) -> list[KvInputSpec]:
    input_meta = {m.name: m for m in session.get_inputs()}
    output_meta = {m.name: m for m in session.get_outputs()}

    key_layers: dict[int, str] = {}
    value_layers: dict[int, str] = {}
    pkey_layers: dict[int, str] = {}
    pvalue_layers: dict[int, str] = {}

    for name in input_meta:
        m = PAST_KEY_RE.match(name)
        if m:
            key_layers[int(m.group(1))] = name
            continue
        m = PAST_VALUE_RE.match(name)
        if m:
            value_layers[int(m.group(1))] = name

    for name in output_meta:
        m = PRESENT_KEY_RE.match(name)
        if m:
            pkey_layers[int(m.group(1))] = name
            continue
        m = PRESENT_VALUE_RE.match(name)
        if m:
            pvalue_layers[int(m.group(1))] = name

    layers = sorted(set(key_layers) & set(value_layers) & set(pkey_layers) & set(pvalue_layers))
    specs: list[KvInputSpec] = []
    for layer in layers:
        key_meta = input_meta[key_layers[layer]]
        shape = key_meta.shape
        if len(shape) != 4:
            raise ValueError(f"Expected rank-4 KV input for {key_meta.name}, got shape {shape}")

        # KV shape convention: [batch, num_kv_heads, seq, head_size]
        num_kv_heads = int(shape[1]) if isinstance(shape[1], int) else 1
        head_size = int(shape[3]) if isinstance(shape[3], int) else 0
        if head_size <= 0:
            raise ValueError(f"Could not infer head_size from input shape for {key_meta.name}: {shape}")

        specs.append(
            KvInputSpec(
                key_name=key_layers[layer],
                value_name=value_layers[layer],
                present_key_name=pkey_layers[layer],
                present_value_name=pvalue_layers[layer],
                kv_dtype=ort_type_to_numpy(key_meta.type),
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            )
        )
    return specs


def make_empty_kv(batch_size: int, kv_specs: list[KvInputSpec]) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for spec in kv_specs:
        empty = np.zeros((batch_size, spec.num_kv_heads, 0, spec.head_size), dtype=spec.kv_dtype)
        feeds[spec.key_name] = empty
        feeds[spec.value_name] = empty
    return feeds


def build_prompt(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def main():
    parser = argparse.ArgumentParser(description="Run Gemma-4 text ONNX with manual KV cache using session.run.")
    parser.add_argument("--model_dir", default="examples/python/gemma4-e2b-cpu", help="Directory with model.onnx + tokenizer files.")
    parser.add_argument("--prompt", default="Write a short 4-line poem about beer.", help="User prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 => greedy, >0 => simple sampling.")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=False)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    input_names = {m.name for m in session.get_inputs()}
    if "input_ids" not in input_names or "attention_mask" not in input_names:
        raise RuntimeError("Model must expose 'input_ids' and 'attention_mask' inputs for this script.")

    kv_specs = discover_kv_specs(session)
    if not kv_specs:
        raise RuntimeError("Could not discover KV cache inputs/outputs in model graph.")

    prompt_text = build_prompt(tokenizer, args.prompt)
    prompt_ids = tokenizer(prompt_text, return_tensors="np")["input_ids"].astype(np.int64)
    batch_size = prompt_ids.shape[0]
    if batch_size != 1:
        raise RuntimeError("This script currently supports batch_size=1 only.")

    attention_mask = np.ones_like(prompt_ids, dtype=np.int64)
    feeds: dict[str, np.ndarray] = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
    }
    feeds.update(make_empty_kv(batch_size, kv_specs))

    outputs = session.run(None, feeds)
    output_names = [m.name for m in session.get_outputs()]
    output_map = dict(zip(output_names, outputs))
    logits = output_map["logits"]
    next_token = int(np.argmax(logits[0, -1, :]))
    generated = [next_token]

    # Move presents -> past for decode loop.
    past_key_values: dict[str, np.ndarray] = {}
    for spec in kv_specs:
        past_key_values[spec.key_name] = output_map[spec.present_key_name]
        past_key_values[spec.value_name] = output_map[spec.present_value_name]

    eos_id = tokenizer.eos_token_id
    for _ in range(args.max_new_tokens - 1):
        if eos_id is not None and next_token == eos_id:
            break

        input_ids = np.array([[next_token]], dtype=np.int64)
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        feeds.update(past_key_values)

        outputs = session.run(None, feeds)
        output_map = dict(zip(output_names, outputs))
        logits = output_map["logits"]
        if args.temperature and args.temperature > 0:
            probs = np.exp(logits[0, -1, :] / args.temperature)
            probs = probs / probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)

        for spec in kv_specs:
            past_key_values[spec.key_name] = output_map[spec.present_key_name]
            past_key_values[spec.value_name] = output_map[spec.present_value_name]

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("Prompt:")
    print(args.prompt)
    print("\nGenerated:")
    print(text.strip())


if __name__ == "__main__":
    main()

