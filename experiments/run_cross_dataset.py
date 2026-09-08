"""Profile each corpus and test portability of the HotpotQA configuration."""

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
from core.partial_prefill import greedy_decode, partial_prefill
from data.jsonl import read_jsonl
from evaluation.metrics import batch_f1_score
from experiments.common import receiver_prefill, run_method
from kv_cache.layer_kv_extractor import extract_layer_caches
from profiling.profiler import candidate_groups, pareto_frontier


def evaluate_configuration(receiver, receiver_tokenizer, prepared, layers, max_new_tokens, device):
    predictions = []
    for sender_kv, sender_e, inputs, _ in prepared:
        prediction, _ = run_method(lambda: partial_prefill(receiver, sender_kv, sender_e, layers, **inputs), receiver, receiver_tokenizer, max_new_tokens, device)
        predictions.append(prediction)
    return batch_f1_score(predictions, [item[3] for item in prepared])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", default=resolve_model_path("DROID_SPEAK_SENDER", DEFAULT_SENDER_PATH))
    parser.add_argument("--receiver", default=resolve_model_path("DROID_SPEAK_RECEIVER", DEFAULT_RECEIVER_PATH))
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--hotpot-profile", default="profiling_results.json")
    parser.add_argument("--output", default="results/cross_dataset_robustness.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    sender, sender_tokenizer = load_model_and_tokenizer(args.sender, args.device)
    receiver, receiver_tokenizer = load_model_and_tokenizer(args.receiver, args.device)
    supplied_hotpot_layers = tuple(json.loads(Path(args.hotpot_profile).read_text(encoding="utf-8"))["optimal"]["layers"])
    datasets = {"hotpotqa": "hotpotqa_train.jsonl", "2wikimqa": "2wikimqa_test.jsonl", "multifieldqa_en": "multifieldqa_en_test.jsonl"}
    output = {"hotpot_profile_layers": list(supplied_hotpot_layers), "datasets": {}}

    for name, filename in datasets.items():
        records = list(read_jsonl(Path(args.data_dir) / filename))[: args.max_samples]
        prepared, baseline_predictions = [], []
        for record in tqdm(records, desc=f"Preparing {name}"):
            sender_inputs = tokenize_prompt(sender_tokenizer, record, args.device)
            receiver_inputs = tokenize_prompt(receiver_tokenizer, record, args.device)
            if not sender_inputs["input_ids"].equal(receiver_inputs["input_ids"]):
                raise ValueError("Sender and receiver token IDs differ")
            sender_kv, sender_e = extract_layer_caches(sender, **sender_inputs)
            prepared.append((sender_kv, sender_e, receiver_inputs, record["answer"]))
            prediction, _ = run_method(lambda: receiver_prefill(receiver, receiver_inputs), receiver, receiver_tokenizer, args.max_new_tokens, args.device)
            baseline_predictions.append(prediction)
        baseline_f1 = batch_f1_score(baseline_predictions, [item[3] for item in prepared])
        profile = []
        for layers in tqdm(list(candidate_groups(len(receiver.model.layers))), desc=f"Profiling {name}"):
            f1 = evaluate_configuration(receiver, receiver_tokenizer, prepared, layers, args.max_new_tokens, args.device)
            profile.append({"layers": list(layers), "recomputed_layers": layers[1] - layers[0] + 1, "f1": f1})
        frontier = pareto_frontier(profile)
        viable = [item for item in frontier if item["f1"] >= baseline_f1 * 0.95]
        own_optimal = min(viable, key=lambda item: item["recomputed_layers"]) if viable else max(frontier, key=lambda item: item["f1"])
        hotpot_f1 = evaluate_configuration(receiver, receiver_tokenizer, prepared, supplied_hotpot_layers, args.max_new_tokens, args.device)
        output["datasets"][name] = {
            "baseline_f1": baseline_f1,
            "dataset_optimal": own_optimal,
            "pareto_frontier": frontier,
            "hotpot_profile_f1": hotpot_f1,
            "hotpot_profile_drop_points": (baseline_f1 - hotpot_f1) * 100,
            "within_four_points": (baseline_f1 - hotpot_f1) <= 0.04,
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
