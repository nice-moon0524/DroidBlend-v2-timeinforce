"""Run the four fixed token-anchor ablations without changing legacy runners.

The experiment keeps each locked DroidSpeak Pareto span dense and uses a
CacheBlend-style sparse correction outside that span.  It never calls receiver
full prefill; receiver-full values in the lock are report-only references.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import torch
from tqdm import tqdm

from core.model_loading import DEFAULT_RECEIVER_PATH, DEFAULT_SENDER_PATH, load_model_and_tokenizer, resolve_model_path, tokenize_prompt
from core.partial_prefill import greedy_decode
from core.token_anchor_sparse_prefill import TokenAnchorPrefillResult, token_anchor_sparse_prefill
from data.jsonl import read_jsonl
from evaluation.metrics import batch_f1_score
from experiments.common import synchronize
from kv_cache.cache_utils import LayerKV
from kv_cache.layer_kv_extractor import extract_layer_caches


DATASET_NAME = "hotpotqa_50"
DATASET_FILE = "hotpotqa_train.jsonl"
DEFAULT_RATIO = 0.20
GROUPS = {
    "layer0-k": ("layer0", "k"),
    "layer0-kv": ("layer0", "kv"),
    "start-k": ("start", "k"),
    "start-kv": ("start", "kv"),
}


def _read_lock(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"DroidSpeak baseline lock is missing: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    spans = payload.get("pareto_spans")
    if not isinstance(spans, list) or len(spans) != 16:
        raise ValueError("baseline lock must contain exactly 16 Pareto spans for this experiment")
    seen: set[tuple[int, int]] = set()
    for item in spans:
        span = item.get("span")
        if not isinstance(span, list) or len(span) != 2 or "baseline_f1" not in item or not item.get("method"):
            raise ValueError(f"invalid baseline-lock item: {item}")
        key = int(span[0]), int(span[1])
        if key in seen:
            raise ValueError(f"duplicate Pareto span in baseline lock: {key}")
        seen.add(key)
    return payload, spans


def _parse_groups(value: str) -> list[str]:
    groups = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not groups or any(group not in GROUPS for group in groups):
        choices = ", ".join(GROUPS)
        raise ValueError(f"--groups must contain only: {choices}")
    return groups


def _ratio_slug(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def _method_name(group: str, start: int, end: int, ratio: float) -> str:
    anchor_mode, score_mode = GROUPS[group]
    return f"token_anchor_{anchor_mode}_score_{score_mode}_layers_{start:02d}-{end:02d}_ratio_{_ratio_slug(ratio)}"


def _move_layer_kv(layer_kv: Mapping[int, LayerKV], device: str, dtype: torch.dtype) -> dict[int, LayerKV]:
    return {
        layer_id: (key.to(device=device, dtype=dtype, non_blocking=True), value.to(device=device, dtype=dtype, non_blocking=True))
        for layer_id, (key, value) in layer_kv.items()
    }


def _move_hidden_cache(cache: Mapping[int, torch.Tensor], device: str, dtype: torch.dtype) -> dict[int, torch.Tensor]:
    return {layer_id: hidden.to(device=device, dtype=dtype, non_blocking=True) for layer_id, hidden in cache.items()}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _timed_prefill(method: Callable[[], TokenAnchorPrefillResult], receiver, tokenizer, max_new_tokens: int, device: str) -> tuple[str, float, dict[str, Any]]:
    synchronize(device)
    started = perf_counter()
    result = method()
    synchronize(device)
    latency = perf_counter() - started
    prediction = greedy_decode(receiver, tokenizer, result, max_new_tokens)
    selection = result.selection
    return prediction, latency, {
        "selection_anchor_layer": selection.selection_anchor_layer,
        "selection_score_mode": selection.selection_score_mode,
        "token_recompute_ratio": selection.token_recompute_ratio,
        "selected_count": selection.selected_count,
        "candidate_count": selection.candidate_count,
        "selected_fraction": selection.selected_count / selection.candidate_count,
        "selected_indices": selection.selected_indices,
        "candidate_indices": selection.candidate_indices,
        "score_k": selection.score_k,
        "score_v": selection.score_v,
        "recompute_layers": list(result.recompute_layers),
        "span_mode": "contiguous",
        "full_recompute_layer_count": result.full_recompute_layer_count,
        "sparse_recompute_layer_count": result.sparse_recompute_layer_count,
        "recompute_work_ratio": result.recompute_work_ratio,
        "receiver_full_dependency": result.receiver_full_dependency,
    }


def _aggregate(rows: list[dict[str, Any]], references: list[Any]) -> dict[str, Any]:
    latencies = [float(row["prefill_latency_s"]) for row in rows]
    total_selected = sum(int(row["metadata"]["selected_count"]) for row in rows)
    total_candidates = sum(int(row["metadata"]["candidate_count"]) for row in rows)
    parameters = dict(rows[0]["metadata"]) if rows else {}
    for key in ("selected_indices", "candidate_indices", "score_k", "score_v"):
        parameters.pop(key, None)
    if total_candidates:
        parameters["selected_count"] = total_selected
        parameters["candidate_count"] = total_candidates
        parameters["selected_fraction"] = total_selected / total_candidates
        parameters["recompute_work_ratio"] = (
            parameters["full_recompute_layer_count"] * total_candidates
            + parameters["sparse_recompute_layer_count"] * total_selected
        ) / (parameters["full_recompute_layer_count"] + parameters["sparse_recompute_layer_count"]) / total_candidates
    return {
        "f1": batch_f1_score([row["prediction"] for row in rows], references),
        "prefill_latency_s_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "prefill_latency_s_p50": _percentile(latencies, 0.50),
        "prefill_latency_s_p95": _percentile(latencies, 0.95),
        "samples": len(rows),
        "parameters": parameters,
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for method, values in summary["methods"].items():
        row = {"dataset": summary["dataset"], "method": method, **{key: value for key, value in values.items() if key != "parameters"}}
        row.update({f"param_{key}": value for key, value in values["parameters"].items()})
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _validate_offline(args: argparse.Namespace) -> None:
    lock, spans = _read_lock(args.baseline_lock)
    data_path = Path(args.data_dir) / DATASET_FILE
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if args.token_recompute_ratio != DEFAULT_RATIO:
        raise ValueError("this first ablation is intentionally fixed at token_recompute_ratio=0.20")
    groups = _parse_groups(args.groups)
    print(json.dumps({
        "status": "offline_validation_passed",
        "dataset": lock.get("dataset"),
        "pareto_span_count": len(spans),
        "groups": groups,
        "token_recompute_ratio": args.token_recompute_ratio,
        "receiver_full_dependency": False,
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-group token-anchor K/KV ablation on locked DroidSpeak Pareto spans")
    parser.add_argument("--sender", default=resolve_model_path("DROIDBLEND_SENDER", DEFAULT_SENDER_PATH))
    parser.add_argument("--receiver", default=resolve_model_path("DROIDBLEND_RECEIVER", DEFAULT_RECEIVER_PATH))
    parser.add_argument("--baseline-lock", default="docs/droidspeak_pareto_baseline_lock.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/token_anchor_ablation/hotpotqa_50_ratio_0p20")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sender-device", default="")
    parser.add_argument("--receiver-device", default="")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--token-recompute-ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline-validate", action="store_true")
    args = parser.parse_args()

    if args.offline_validate:
        _validate_offline(args)
        return
    if args.max_samples != 50:
        raise ValueError("the first experiment must use exactly --max-samples 50 so all four groups share hotpotqa_50")
    if args.token_recompute_ratio != DEFAULT_RATIO:
        raise ValueError("the first experiment is fixed at --token-recompute-ratio 0.20")
    groups = _parse_groups(args.groups)
    lock, lock_items = _read_lock(args.baseline_lock)
    data_path = Path(args.data_dir) / DATASET_FILE
    records = list(read_jsonl(data_path))[:args.max_samples]
    if len(records) != args.max_samples:
        raise ValueError(f"{data_path} contains only {len(records)} records, expected 50")

    receiver_device = args.receiver_device or args.device
    sender_device = args.sender_device or receiver_device
    sender, _ = load_model_and_tokenizer(args.sender, sender_device, args.dtype, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    receiver, tokenizer = load_model_and_tokenizer(args.receiver, receiver_device, args.dtype, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    receiver_dtype = next(receiver.parameters()).dtype
    receiver_layers = len(receiver.model.layers)
    for item in lock_items:
        start, end = map(int, item["span"])
        if not 0 <= start <= end < receiver_layers:
            raise ValueError(f"locked span {(start, end)} is outside this receiver ({receiver_layers} layers)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    example_path = output_dir / "per_example.jsonl"
    with example_path.open("w", encoding="utf-8") as stream:
        for record in tqdm(records, desc=DATASET_NAME):
            receiver_inputs = tokenize_prompt(tokenizer, record, receiver_device, args.max_prompt_tokens)
            sender_inputs = {key: value.to(sender_device) if hasattr(value, "to") else value for key, value in receiver_inputs.items()}
            sender_kv, sender_e = extract_layer_caches(sender, **sender_inputs, include_e_cache=True)
            sender_kv = _move_layer_kv(sender_kv, receiver_device, receiver_dtype)
            sender_e = _move_hidden_cache(sender_e, receiver_device, receiver_dtype)
            sample = {"dataset": DATASET_NAME, "id": record.get("id"), "sequence_length": int(receiver_inputs["input_ids"].shape[1]), "methods": {}}
            for item in lock_items:
                start, end = map(int, item["span"])
                for group in groups:
                    anchor_mode, score_mode = GROUPS[group]
                    anchor = 0 if anchor_mode == "layer0" else start
                    method = _method_name(group, start, end, args.token_recompute_ratio)
                    prediction, latency, metadata = _timed_prefill(
                        lambda start=start, end=end, anchor=anchor, score_mode=score_mode: token_anchor_sparse_prefill(
                            receiver, sender_kv, sender_e, receiver_inputs["input_ids"], receiver_inputs.get("attention_mask"),
                            (start, end), anchor, score_mode, args.token_recompute_ratio,
                        ),
                        receiver, tokenizer, args.max_new_tokens, receiver_device,
                    )
                    metadata.update({
                        "anchor_mode": anchor_mode,
                        "droidspeak_baseline_method": item["method"],
                        "droidspeak_baseline_f1": item["baseline_f1"],
                        "droidspeak_baseline_prefill_latency_s_mean": item.get("baseline_prefill_latency_s_mean"),
                    })
                    row = {"prediction": prediction, "prefill_latency_s": latency, "metadata": metadata}
                    rows_by_method[method].append(row)
                    sample["methods"][method] = {"prediction": prediction, "prefill_latency_s": latency, **metadata}
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stream.flush()
            del sender_kv, sender_e, sender_inputs, receiver_inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    methods = {method: _aggregate(rows, [record["answer"] for record in records]) for method, rows in rows_by_method.items()}
    for values in methods.values():
        parameters = values["parameters"]
        values["f1_vs_droidspeak"] = values["f1"] - parameters["droidspeak_baseline_f1"]
        baseline_latency = parameters["droidspeak_baseline_prefill_latency_s_mean"]
        values["latency_vs_droidspeak"] = None if baseline_latency is None else values["prefill_latency_s_mean"] - baseline_latency
        values["f1_vs_receiver_full"] = values["f1"] - float(lock["receiver_full_f1"])
        values["receiver_full_dependency"] = False
    summary = {
        "project": "DroidBlend-v2-timeinforce",
        "experiment": "token_anchor_k_kv_ablation_v1",
        "dataset": DATASET_NAME,
        "samples": args.max_samples,
        "receiver_full_dependency": False,
        "latency_scope": "selection projection + dense DroidSpeak span + sparse token correction + KV merge + final prompt logit; excludes sender cache extraction and greedy decode",
        "baseline_lock": str(args.baseline_lock),
        "methods": methods,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "summary.csv", summary)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "summary.json"), "csv": str(output_dir / "summary.csv"), "per_example": str(example_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
