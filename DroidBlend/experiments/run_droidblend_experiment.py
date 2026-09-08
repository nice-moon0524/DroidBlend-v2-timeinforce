"""Run the DroidBlend reference experiment on QA datasets.

The experiment consumes an existing DroidSpeak profiling result and does not
enumerate layer groups itself.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from core.droidblend_prefill import DroidBlendPrefillResult, droidblend_reference_prefill
from core.model_loading import (
    DEFAULT_RECEIVER_PATH,
    DEFAULT_SENDER_PATH,
    load_model_and_tokenizer,
    make_prompt,
    resolve_model_path,
    tokenize_prompt,
)
from core.partial_prefill import greedy_decode, partial_prefill
from data.jsonl import read_jsonl
from evaluation.metrics import batch_f1_score
from experiments.common import cacheblend_prefill, full_reuse_prefill, receiver_prefill, run_method, synchronize
from kv_cache.layer_kv_extractor import extract_layer_caches


def _read_recompute_layers(path: str | Path) -> tuple[int, int]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"DroidSpeak profiling result is missing: {profile_path}. "
            "Run DroidSpeak-new profiling first or pass --profiling-results."
        )
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    layers = raw.get("optimal", {}).get("layers")
    if not isinstance(layers, list) or len(layers) != 2:
        raise ValueError(f"{profile_path} does not contain optimal.layers = [start, end]")
    return int(layers[0]), int(layers[1])


def _assert_same_tokenization(sender_tokenizer, receiver_tokenizer, record: dict, device: str):
    sender_inputs = tokenize_prompt(sender_tokenizer, record, device)
    receiver_inputs = tokenize_prompt(receiver_tokenizer, record, device)
    if not sender_inputs["input_ids"].equal(receiver_inputs["input_ids"]):
        raise ValueError("Sender and receiver tokenizers produced different token ids for the same prompt")
    return sender_inputs, receiver_inputs


def _timed_droidblend(method, receiver, receiver_tokenizer, max_new_tokens: int, device: str):
    synchronize(device)
    started = perf_counter()
    result: DroidBlendPrefillResult = method()
    synchronize(device)
    latency = perf_counter() - started
    prediction = greedy_decode(receiver, receiver_tokenizer, result, max_new_tokens)
    return prediction, latency, result.selection


def _excerpt(text: str, limit: int = 800) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", default=resolve_model_path("DROIDBLEND_SENDER", DEFAULT_SENDER_PATH))
    parser.add_argument("--receiver", default=resolve_model_path("DROIDBLEND_RECEIVER", DEFAULT_RECEIVER_PATH))
    parser.add_argument("--profiling-results", default="../DroidSpeak-new/profiling_results.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/droidblend_quality_latency.json")
    parser.add_argument("--token-output", default="results/droidblend_token_selection.json")
    parser.add_argument("--examples-output", default="results/droidblend_examples.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--important-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--selection-layer", type=int, default=0)
    args = parser.parse_args()

    recompute_layers = _read_recompute_layers(args.profiling_results)
    sender, sender_tokenizer = load_model_and_tokenizer(args.sender, args.device)
    receiver, receiver_tokenizer = load_model_and_tokenizer(args.receiver, args.device)
    datasets = {
        "hotpotqa": "hotpotqa_test.jsonl",
        "2wikimqa": "2wikimqa_test.jsonl",
        "multifieldqa_en": "multifieldqa_en_test.jsonl",
    }
    report = {
        "method": "droidblend_reference",
        "reference_mode": True,
        "reference_mode_note": (
            "DroidBlend and cacheblend-style methods use receiver full prefill as a correctness reference "
            "because stock Transformers lacks efficient arbitrary-token sparse prefill kernels."
        ),
        "profiling_results": str(args.profiling_results),
        "recompute_layers": list(recompute_layers),
        "important_fraction": args.important_fraction,
        "top_k": args.top_k,
        "selection_layer": args.selection_layer,
        "methods": {},
    }
    examples_report = {
        "method": "droidblend_reference",
        "reference_mode": True,
        "profiling_results": str(args.profiling_results),
        "recompute_layers": list(recompute_layers),
        "important_fraction": args.important_fraction,
        "top_k": args.top_k,
        "selection_layer": args.selection_layer,
        "methods": [
            "full_prefill",
            "full_kv_reuse",
            "droidspeak",
            "cacheblend_reference",
            "droidblend_reference",
        ],
        "datasets": {},
    }
    token_report = {
        "recompute_layers": list(recompute_layers),
        "important_fraction": args.important_fraction,
        "top_k": args.top_k,
        "selection_layer": args.selection_layer,
        "datasets": {},
    }

    for name, filename in datasets.items():
        records = list(read_jsonl(Path(args.data_dir) / filename))[: args.max_samples]
        answers: dict[str, list[str]] = defaultdict(list)
        latencies: dict[str, list[float]] = defaultdict(list)
        selections = []
        example_rows = []
        for record in tqdm(records, desc=name):
            sender_inputs, receiver_inputs = _assert_same_tokenization(sender_tokenizer, receiver_tokenizer, record, args.device)
            sender_kv, sender_e = extract_layer_caches(sender, **sender_inputs)
            sample_report = {
                "id": record.get("id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "prompt": make_prompt(record),
                "context_excerpt": _excerpt(str(record.get("context", "")), 600),
                "methods": {},
            }
            methods = {
                "full_prefill": lambda: receiver_prefill(receiver, receiver_inputs),
                "full_kv_reuse": lambda: full_reuse_prefill(receiver, sender_kv, sender_e, **receiver_inputs),
                "droidspeak": lambda: partial_prefill(receiver, sender_kv, sender_e, recompute_layers, **receiver_inputs),
                "cacheblend_reference": lambda: cacheblend_prefill(
                    receiver,
                    sender_kv,
                    receiver_prefill(receiver, receiver_inputs),
                    args.important_fraction,
                    **receiver_inputs,
                ),
            }
            for method_name, method in methods.items():
                prediction, latency = run_method(method, receiver, receiver_tokenizer, args.max_new_tokens, args.device)
                answers[method_name].append(prediction)
                latencies[method_name].append(latency)
                sample_report["methods"][method_name] = {
                    "prediction": prediction,
                    "latency_s": latency,
                }

            prediction, latency, selection = _timed_droidblend(
                lambda: droidblend_reference_prefill(
                    receiver,
                    sender_kv,
                    sender_e,
                    recompute_layers,
                    important_fraction=args.important_fraction,
                    top_k=args.top_k,
                    selection_layer=args.selection_layer,
                    **receiver_inputs,
                ),
                receiver,
                receiver_tokenizer,
                args.max_new_tokens,
                args.device,
            )
            answers["droidblend_reference"].append(prediction)
            latencies["droidblend_reference"].append(latency)
            sample_report["methods"]["droidblend_reference"] = {
                "prediction": prediction,
                "latency_s": latency,
                "selection": {
                    "selected_count": selection.selected_count,
                    "candidate_count": selection.candidate_count,
                    "selected_fraction": selection.selected_count / selection.candidate_count,
                    "indices": selection.indices,
                },
            }
            selections.append(
                {
                    "id": record.get("id"),
                    "selected_count": selection.selected_count,
                    "candidate_count": selection.candidate_count,
                    "selected_fraction": selection.selected_count / selection.candidate_count,
                    "indices": selection.indices,
                }
            )
            example_rows.append(sample_report)

        references = [record["answer"] for record in records]
        report["methods"][name] = {
            method: {
                "f1": batch_f1_score(predictions, references),
                "prefill_latency_s": sum(latencies[method]) / len(latencies[method]),
                "samples": len(records),
            }
            for method, predictions in answers.items()
        }
        token_report["datasets"][name] = {
            "samples": len(records),
            "average_selected_count": sum(item["selected_count"] for item in selections) / len(selections) if selections else 0,
            "average_candidate_count": sum(item["candidate_count"] for item in selections) / len(selections) if selections else 0,
            "records": selections,
        }
        examples_report["datasets"][name] = {
            "samples": len(example_rows),
            "records": example_rows,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.token_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.token_output).write_text(json.dumps(token_report, indent=2), encoding="utf-8")
    Path(args.examples_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.examples_output).write_text(json.dumps(examples_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
