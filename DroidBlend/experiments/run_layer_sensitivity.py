"""Measure quality loss from replacing exactly one decoder layer's KV cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from core.model_loading import (
    DEFAULT_RECEIVER_PATH,
    DEFAULT_SENDER_PATH,
    load_model_and_tokenizer,
    resolve_model_path,
    tokenize_prompt,
)
from core.partial_prefill import single_layer_reuse_prefill
from data.jsonl import read_jsonl
from evaluation.metrics import batch_f1_score
from experiments.common import receiver_prefill, run_method
from kv_cache.layer_kv_extractor import extract_layer_caches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", default=resolve_model_path("DROID_SPEAK_SENDER", DEFAULT_SENDER_PATH))
    parser.add_argument("--receiver", default=resolve_model_path("DROID_SPEAK_RECEIVER", DEFAULT_RECEIVER_PATH))
    parser.add_argument("--dataset", default="data/processed/hotpotqa_test.jsonl")
    parser.add_argument("--output", default="results/layer_sensitivity.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=250)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    sender, sender_tokenizer = load_model_and_tokenizer(args.sender, args.device)
    receiver, receiver_tokenizer = load_model_and_tokenizer(args.receiver, args.device)
    prepared = []
    baseline = []
    for record in tqdm(list(read_jsonl(args.dataset))[: args.max_samples], desc="Preparing samples"):
        sender_inputs = tokenize_prompt(sender_tokenizer, record, args.device)
        receiver_inputs = tokenize_prompt(receiver_tokenizer, record, args.device)
        if not sender_inputs["input_ids"].equal(receiver_inputs["input_ids"]):
            raise ValueError("Sender and receiver token IDs differ")
        sender_kv, sender_e = extract_layer_caches(sender, **sender_inputs)
        prepared.append((sender_kv, sender_e, receiver_inputs, record["answer"]))
        prediction, _ = run_method(lambda: receiver_prefill(receiver, receiver_inputs), receiver, receiver_tokenizer, args.max_new_tokens, args.device)
        baseline.append(prediction)
    baseline_f1 = batch_f1_score(baseline, [item[3] for item in prepared])
    layers = []
    for layer_id in tqdm(range(len(receiver.model.layers)), desc="Layer sensitivity"):
        predictions = []
        for sender_kv, sender_e, inputs, _ in prepared:
            prediction, _ = run_method(lambda: single_layer_reuse_prefill(receiver, sender_kv, sender_e, layer_id, **inputs), receiver, receiver_tokenizer, args.max_new_tokens, args.device)
            predictions.append(prediction)
        f1 = batch_f1_score(predictions, [item[3] for item in prepared])
        layers.append({"layer": layer_id, "f1": f1, "f1_drop": baseline_f1 - f1, "critical": baseline_f1 > 0 and (baseline_f1 - f1) / baseline_f1 > 0.10})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"baseline_f1": baseline_f1, "layers": layers, "critical_layers": [item["layer"] for item in layers if item["critical"]]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
