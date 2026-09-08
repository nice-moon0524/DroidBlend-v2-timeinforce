"""Run DroidSpeak baselines and anchor-based sparse DroidBlend experiments."""

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
from core.partial_prefill import greedy_decode, partial_prefill
from core.sparse_droidblend import SparseDroidBlendResult, sparse_droidblend_prefill
from data.jsonl import read_jsonl
from evaluation.metrics import batch_f1_score
from experiments.common import full_reuse_prefill, receiver_prefill, synchronize
from kv_cache.cache_utils import LayerKV
from kv_cache.layer_kv_extractor import extract_layer_caches


DATASETS = {"hotpotqa_50": "hotpotqa_train.jsonl"}
DEFAULT_TOKEN_RATIOS = "0.10,0.20,0.30,0.40,0.50"


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value <= 0 or value > 1 for value in values):
        raise ValueError("token recompute ratios must be in (0, 1]")
    return list(dict.fromkeys(values))


def _parse_span(value: str) -> tuple[int, int]:
    parts = [part for part in value.strip().replace(":", "-").replace("_", "-").split("-") if part]
    if len(parts) != 2:
        raise ValueError(f"invalid recompute span '{value}', expected start-end")
    start, end = map(int, parts)
    if start > end:
        raise ValueError(f"invalid recompute span '{value}'")
    return start, end


def _read_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"DroidSpeak profile is missing: {profile_path}")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    frontier = raw.get("pareto_frontier")
    if not isinstance(frontier, list) or not frontier:
        optimal = raw.get("optimal")
        if isinstance(optimal, dict) and isinstance(optimal.get("layers"), list):
            frontier = [optimal]
        else:
            raise ValueError(f"{profile_path} does not contain pareto_frontier")
    raw["pareto_frontier"] = frontier
    return raw


def _read_baseline_lock(path: str | Path) -> dict[tuple[int, int], dict[str, Any]]:
    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"DroidSpeak baseline lock is missing: {lock_path}")
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    items = raw.get("pareto_spans")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{lock_path} does not contain pareto_spans")
    baselines: dict[tuple[int, int], dict[str, Any]] = {}
    for item in items:
        span = item.get("span")
        if not isinstance(span, list) or len(span) != 2 or not item.get("method") or "baseline_f1" not in item:
            raise ValueError(f"invalid locked DroidSpeak baseline: {item}")
        key = (int(span[0]), int(span[1]))
        if key in baselines:
            raise ValueError(f"duplicate locked DroidSpeak span: {key}")
        baselines[key] = item
    return baselines


def _resolve_spans(value: str, profile: dict[str, Any], baseline_lock: dict[tuple[int, int], dict[str, Any]]) -> list[tuple[int, int]]:
    if value == "pareto":
        spans = [tuple(map(int, item["span"])) for item in baseline_lock.values()]
    elif value == "profile":
        spans = [tuple(map(int, item["layers"])) for item in profile["pareto_frontier"]]
    else:
        spans = [_parse_span(item) for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(spans))


def _parse_anchors(value: str, start: int) -> list[int]:
    anchors = [start if item.strip() == "start" else int(item.strip()) for item in value.split(",") if item.strip()]
    if not anchors or any(anchor not in {0, start} for anchor in anchors):
        raise ValueError("selection_anchor_layer must contain only 0 and/or start")
    return list(dict.fromkeys(anchors))


def _parse_score_modes(value: str) -> list[str]:
    modes = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not modes or any(mode not in {"k", "kv"} for mode in modes):
        raise ValueError("selection_score_mode must contain only k and/or kv")
    return modes


def _method_name(kind: str, **kwargs: Any) -> str:
    if kind == "droidspeak":
        start, end = kwargs["layers"]
        return f"droidspeak_layers_{start:02d}_{end:02d}"
    if kind == "hybrid":
        start, end = kwargs["layers"]
        ratio = f"{kwargs['ratio']:.3f}".replace(".", "p")
        return f"droidblend_{kwargs['span_mode']}_layers_{start:02d}-{end:02d}_anchor_L{kwargs['anchor']}_score_{kwargs['score_mode']}_ratio_{ratio}"
    return kind


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _move_layer_kv(layer_kv: Mapping[int, LayerKV], device: str, dtype: torch.dtype) -> dict[int, LayerKV]:
    return {layer: (key.to(device=device, dtype=dtype, non_blocking=True), value.to(device=device, dtype=dtype, non_blocking=True)) for layer, (key, value) in layer_kv.items()}


def _move_hidden_cache(hidden_cache: Mapping[int, torch.Tensor], device: str, dtype: torch.dtype) -> dict[int, torch.Tensor]:
    return {layer: hidden.to(device=device, dtype=dtype, non_blocking=True) for layer, hidden in hidden_cache.items()}


def _timed(method: Callable[[], Any], receiver, tokenizer, max_new_tokens: int, device: str) -> tuple[str, float, dict[str, Any]]:
    synchronize(device)
    started = perf_counter()
    result = method()
    synchronize(device)
    elapsed = perf_counter() - started
    prediction = greedy_decode(receiver, tokenizer, result, max_new_tokens)
    metadata: dict[str, Any] = {}
    if isinstance(result, SparseDroidBlendResult):
        selection = result.selection
        metadata = {
            "selection_anchor_layer": selection.selection_anchor_layer,
            "selection_score_mode": selection.selection_score_mode,
            "token_recompute_ratio": selection.token_recompute_ratio,
            "selected_count": selection.selected_count,
            "candidate_count": selection.candidate_count,
            "selected_fraction": selection.selected_count / selection.candidate_count,
            "selected_indices": selection.indices,
            "candidate_indices": selection.candidate_indices,
            "recompute_layers": list(result.recompute_layers),
            "span_mode": result.span_mode,
            "full_recompute_layer_count": result.full_recompute_layer_count,
            "recompute_work_ratio": result.recompute_work_ratio,
            "receiver_full_dependency": result.receiver_full_dependency,
        }
    return prediction, elapsed, metadata


def _aggregate(rows: list[dict[str, Any]], references: list[Any]) -> dict[str, Any]:
    latencies = [float(row["latency_s"]) for row in rows]
    return {
        "f1": batch_f1_score([row["prediction"] for row in rows], references),
        "prefill_latency_s_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "prefill_latency_s_p50": _percentile(latencies, 0.50),
        "prefill_latency_s_p95": _percentile(latencies, 0.95),
        "samples": len(rows),
        "parameters": rows[0]["parameters"] if rows else {},
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for dataset, methods in summary["datasets"].items():
        for method, values in methods.items():
            row = {"dataset": dataset, "method": method, **{key: value for key, value in values.items() if key != "parameters"}}
            row.update({f"param_{key}": value for key, value in values.get("parameters", {}).items()})
            rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="DroidSpeak-sensitive-layer and sparse-token experiments")
    parser.add_argument("--sender", default=resolve_model_path("DROIDBLEND_SENDER", DEFAULT_SENDER_PATH))
    parser.add_argument("--receiver", default=resolve_model_path("DROIDBLEND_RECEIVER", DEFAULT_RECEIVER_PATH))
    parser.add_argument("--profiling-results", default="../DroidSpeak-new/profiling_results.json")
    parser.add_argument("--baseline-lock", default="docs/droidspeak_pareto_baseline_lock.json")
    parser.add_argument("--validate-baseline-lock", action="store_true")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/hybrid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sender-device", default="")
    parser.add_argument("--receiver-device", default="")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--mode", choices=["baselines", "sweep", "all"], default="all")
    parser.add_argument("--recompute-spans", "--recompute-layers", dest="recompute_spans", default="pareto")
    parser.add_argument("--recompute-layers-file", default="", help="baseline-lock JSON whose pareto spans should be swept")
    parser.add_argument("--token-recompute-ratio", default=None)
    parser.add_argument("--token-ratios", default=None)
    parser.add_argument("--selection-anchor-layer", default="0")
    parser.add_argument("--selection-score-mode", default="k")
    parser.add_argument("--span-mode", choices=["contiguous", "disjoint"], default="contiguous")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--no-full-prefill-baseline", action="store_true")
    args = parser.parse_args()

    if args.span_mode != "contiguous":
        raise NotImplementedError("span_mode=disjoint is reserved for a later experiment phase")
    baseline_lock = _read_baseline_lock(args.baseline_lock)
    if args.validate_baseline_lock:
        print(json.dumps({"status": "baseline_lock_valid", "spans": len(baseline_lock)}, ensure_ascii=False))
        return
    profile = _read_profile(args.profiling_results)
    spans = list(_read_baseline_lock(args.recompute_layers_file)) if args.recompute_layers_file else _resolve_spans(args.recompute_spans, profile, baseline_lock)
    if any(span not in baseline_lock for span in spans):
        raise ValueError("every requested recompute layer span must exist in --baseline-lock")
    ratio_text = args.token_recompute_ratio or args.token_ratios or DEFAULT_TOKEN_RATIOS
    token_ratios = _parse_float_list(ratio_text)
    score_modes = _parse_score_modes(args.selection_score_mode)
    if args.offline_only:
        for start, end in spans:
            if (start, end) not in baseline_lock:
                raise ValueError(f"no locked DroidSpeak baseline for span {(start, end)}")
            _parse_anchors(args.selection_anchor_layer, start)
        for filename in DATASETS.values():
            if not (Path(args.data_dir) / filename).is_file():
                raise FileNotFoundError(Path(args.data_dir) / filename)
        print(json.dumps({"status": "offline_validation_passed", "spans": spans, "ratios": token_ratios}, ensure_ascii=False))
        return

    receiver_device = args.receiver_device or args.device
    sender_device = args.sender_device or receiver_device
    sender, _ = load_model_and_tokenizer(args.sender, sender_device, args.dtype, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    receiver, receiver_tokenizer = load_model_and_tokenizer(args.receiver, receiver_device, args.dtype, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    receiver_dtype = next(receiver.parameters()).dtype
    num_layers = len(receiver.model.layers)
    if any(start < 0 or start > end or end >= num_layers for start, end in spans):
        raise ValueError("recompute_layers is outside receiver layer range")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    example_path = output_dir / "per_example.jsonl"
    datasets_summary: dict[str, dict[str, Any]] = {}
    with example_path.open("w", encoding="utf-8") as example_stream:
        for dataset, filename in DATASETS.items():
            records = list(read_jsonl(Path(args.data_dir) / filename))
            if args.max_samples is not None:
                records = records[:args.max_samples]
            rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in tqdm(records, desc=dataset):
                receiver_inputs = tokenize_prompt(receiver_tokenizer, record, receiver_device, args.max_prompt_tokens)
                sender_inputs = {key: value.to(sender_device) if hasattr(value, "to") else value for key, value in receiver_inputs.items()}
                sender_kv, sender_e = extract_layer_caches(sender, **sender_inputs, include_e_cache=True)
                sender_kv = _move_layer_kv(sender_kv, receiver_device, receiver_dtype)
                sender_e = _move_hidden_cache(sender_e, receiver_device, receiver_dtype)
                jobs: list[tuple[str, Callable[[], Any], dict[str, Any]]] = []
                if args.mode in {"baselines", "all"}:
                    if not args.no_full_prefill_baseline:
                        jobs.append(("receiver_full", lambda: receiver_prefill(receiver, receiver_inputs), {"receiver_full_dependency": True}))
                    jobs.append(("full_kv_reuse", lambda: full_reuse_prefill(receiver, sender_kv, sender_e, **receiver_inputs), {"receiver_full_dependency": False}))
                    for start, end in spans:
                        baseline = baseline_lock[(start, end)]
                        jobs.append((_method_name("droidspeak", layers=(start, end)), lambda start=start, end=end: partial_prefill(receiver, sender_kv, sender_e, (start, end), **receiver_inputs), {"receiver_full_dependency": False, "span_mode": "contiguous", "recompute_layers": [start, end], "droidspeak_baseline_method": baseline["method"]}))
                if args.mode in {"sweep", "all"}:
                    for start, end in spans:
                        baseline = baseline_lock[(start, end)]
                        for anchor in _parse_anchors(args.selection_anchor_layer, start):
                            for score_mode in score_modes:
                                for ratio in token_ratios:
                                    name = _method_name("hybrid", layers=(start, end), ratio=ratio, anchor=anchor, score_mode=score_mode, span_mode=args.span_mode)
                                    params = {"receiver_full_dependency": False, "span_mode": args.span_mode, "recompute_layers": [start, end], "selection_anchor_layer": anchor, "selection_score_mode": score_mode, "token_recompute_ratio": ratio, "droidspeak_baseline_method": baseline["method"], "droidspeak_baseline_f1": baseline["baseline_f1"], "droidspeak_baseline_prefill_latency": baseline.get("baseline_prefill_latency_s_mean")}
                                    jobs.append((name, lambda start=start, end=end, anchor=anchor, score_mode=score_mode, ratio=ratio: sparse_droidblend_prefill(receiver, sender_kv, **receiver_inputs, sender_e_cache=sender_e, recompute_layers=(start, end), selection_anchor_layer=anchor, selection_score_mode=score_mode, token_recompute_ratio=ratio, span_mode=args.span_mode), params))
                sample = {"dataset": dataset, "id": record.get("id"), "sequence_length": int(receiver_inputs["input_ids"].shape[1]), "methods": {}}
                for name, job, params in jobs:
                    prediction, latency, metadata = _timed(job, receiver, receiver_tokenizer, args.max_new_tokens, receiver_device)
                    params = {**params, **metadata}
                    rows[name].append({"prediction": prediction, "latency_s": latency, "parameters": params})
                    sample["methods"][name] = {"prediction": prediction, "latency_s": latency, "parameters": params}
                example_stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                example_stream.flush()
                del sender_kv, sender_e, sender_inputs, receiver_inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            methods = {name: _aggregate(method_rows, [record["answer"] for record in records]) for name, method_rows in rows.items()}
            receiver_full = methods.get("receiver_full")
            for name, values in methods.items():
                if not name.startswith("droidblend_contiguous_"):
                    continue
                params = values["parameters"]
                values["f1_vs_droidspeak"] = float(values["f1"]) - float(params["droidspeak_baseline_f1"])
                baseline_latency = params.get("droidspeak_baseline_prefill_latency")
                values["latency_vs_droidspeak"] = None if baseline_latency is None else float(values["prefill_latency_s_mean"]) - float(baseline_latency)
                values["f1_vs_receiver_full"] = None if receiver_full is None else float(values["f1"]) - float(receiver_full["f1"])
                values["latency_vs_receiver_full"] = None if receiver_full is None else float(values["prefill_latency_s_mean"]) - float(receiver_full["prefill_latency_s_mean"])
            datasets_summary[dataset] = methods

    summary = {"project": "DroidBlend-v2-timeinforce", "mode": args.mode, "reference_mode": False, "receiver_full_dependency": False, "hybrid_definition": "Selection computes only local receiver K/V projections at selection_anchor_layer. Full receiver blocks are restricted to recompute_layers; receiver_full is baseline-only.", "baseline_lock": str(args.baseline_lock), "parameters": {"recompute_spans": [list(span) for span in spans], "selection_anchor_layer": args.selection_anchor_layer, "selection_score_mode": score_modes, "token_recompute_ratio": token_ratios, "span_mode": args.span_mode}, "datasets": datasets_summary}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "summary.csv", summary)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "summary.json"), "csv": str(output_dir / "summary.csv"), "examples": str(example_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

