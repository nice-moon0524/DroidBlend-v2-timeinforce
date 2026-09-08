"""Figure-10-style quality/latency comparison across three QA datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from core.model_loading import (
    DEFAULT_RECEIVER_PATH,
    DEFAULT_SENDER_PATH,
    load_model_and_tokenizer,
    resolve_model_path,
    tokenize_prompt,
)
from core.partial_prefill import partial_prefill
from data.jsonl import read_jsonl
from evaluation.metrics import batch_f1_score
from experiments.common import cacheblend_prefill, full_reuse_prefill, receiver_prefill, run_method
from kv_cache.layer_kv_extractor import extract_layer_caches


def _assert_same_tokenization(sender_tokenizer, receiver_tokenizer, record: dict, device: str):
    sender_inputs = tokenize_prompt(sender_tokenizer, record, device)
    receiver_inputs = tokenize_prompt(receiver_tokenizer, record, device)
    if not sender_inputs["input_ids"].equal(receiver_inputs["input_ids"]):
        raise ValueError("Sender and receiver tokenizers produced different token ids for the same prompt")
    return sender_inputs, receiver_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", default=resolve_model_path("DROID_SPEAK_SENDER", DEFAULT_SENDER_PATH))
    parser.add_argument("--receiver", default=resolve_model_path("DROID_SPEAK_RECEIVER", DEFAULT_RECEIVER_PATH))
    parser.add_argument("--profiling-results", default="profiling_results.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/quality_latency_tradeoff.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cacheblend-important-fraction", type=float, default=0.2)
    args = parser.parse_args()
    profile = json.loads(Path(args.profiling_results).read_text(encoding="utf-8"))
    recompute_layers = tuple(profile["optimal"]["layers"])
    sender, sender_tokenizer = load_model_and_tokenizer(args.sender, args.device)
    receiver, receiver_tokenizer = load_model_and_tokenizer(args.receiver, args.device)
    datasets = {"hotpotqa": "hotpotqa_test.jsonl", "2wikimqa": "2wikimqa_test.jsonl", "multifieldqa_en": "multifieldqa_en_test.jsonl"}
    report = {"recompute_layers": list(recompute_layers), "methods": {}}

    for name, filename in datasets.items():
        answers: dict[str, list[str]] = defaultdict(list)
        latencies: dict[str, list[float]] = defaultdict(list)
        records = list(read_jsonl(Path(args.data_dir) / filename))[: args.max_samples]
        for record in tqdm(records, desc=name):
            sender_inputs, receiver_inputs = _assert_same_tokenization(sender_tokenizer, receiver_tokenizer, record, args.device)
            sender_kv, sender_e = extract_layer_caches(sender, **sender_inputs)
            methods = {
                "full_prefill": lambda: receiver_prefill(receiver, receiver_inputs),
                "full_kv_reuse": lambda: full_reuse_prefill(receiver, sender_kv, sender_e, **receiver_inputs),
                "droidspeak": lambda: partial_prefill(receiver, sender_kv, sender_e, recompute_layers, **receiver_inputs),
                "cacheblend": lambda: cacheblend_prefill(receiver, sender_kv, receiver_prefill(receiver, receiver_inputs), args.cacheblend_important_fraction, **receiver_inputs),
            }
            for method_name, method in methods.items():
                prediction, latency = run_method(method, receiver, receiver_tokenizer, args.max_new_tokens, args.device)
                answers[method_name].append(prediction)
                latencies[method_name].append(latency)
        report["methods"][name] = {
            method: {"f1": batch_f1_score(predictions, [record["answer"] for record in records]), "prefill_latency_s": sum(latencies[method]) / len(latencies[method]), "samples": len(records)}
            for method, predictions in answers.items()
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
