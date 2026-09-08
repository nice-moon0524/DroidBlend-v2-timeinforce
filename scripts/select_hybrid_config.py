"""Select reportable DroidBlend configurations from a hybrid summary.

Default rule: keep only hybrid candidates whose F1 is not below the receiver
full-prefill baseline on every reported dataset, then minimize estimated
recompute work.  This matches the project goal: quality should not drop, and
both full-span layer recomputation and token recomputation should be as small as
possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _latency(values: dict[str, Any]) -> float:
    return float(values.get("prefill_latency_s_mean", values.get("prefill_latency_s", 0.0)))


def _candidate_key(method: str, values: dict[str, Any]) -> tuple[float, float, int, int, float, str]:
    params = values.get("parameters", {})
    work = float(params.get("recompute_work_ratio", 1.0))
    ratio = float(params.get("token_recompute_ratio", 1.0))
    layer_count = int(params.get("full_recompute_layer_count", params.get("recompute_layer_count", 9999)))
    span = params.get("recompute_layers", [9999, 9999])
    start = int(span[0]) if isinstance(span, list) and span else 9999
    return work, ratio, layer_count, start, _latency(values), method


def _baseline_f1(methods: dict[str, Any], baseline_method: str) -> float:
    if baseline_method in methods:
        return float(methods[baseline_method]["f1"])
    return max(float(values["f1"]) for values in methods.values())


def _hybrid_methods(methods: dict[str, Any]) -> dict[str, Any]:
    return {method: values for method, values in methods.items() if method.startswith("hybrid_layers_") or method.startswith("hybrid_dense_")}


def _select_per_dataset(methods: dict[str, Any], baseline_method: str, max_f1_drop: float) -> dict[str, Any]:
    baseline = _baseline_f1(methods, baseline_method)
    floor = baseline - max_f1_drop
    hybrids = _hybrid_methods(methods)
    feasible = {
        method: values
        for method, values in hybrids.items()
        if float(values.get("f1", 0.0)) >= floor
    }
    pool = feasible or hybrids
    if not pool:
        return {"status": "no_hybrid_candidates", "baseline_f1": baseline, "quality_floor": floor}
    selected_method = min(pool, key=lambda method: _candidate_key(method, pool[method]))
    selected = pool[selected_method]
    return {
        "status": "feasible" if feasible else "fallback_best_available",
        "baseline_method": baseline_method,
        "baseline_f1": baseline,
        "quality_floor": floor,
        "selected_method": selected_method,
        "selected": selected,
        "f1_delta_vs_baseline": float(selected["f1"]) - baseline,
        "feasible_count": len(feasible),
        "hybrid_candidate_count": len(hybrids),
    }


def _select_global(summary: dict[str, Any], baseline_method: str, max_f1_drop: float) -> dict[str, Any]:
    datasets = summary.get("datasets", {})
    if not datasets:
        return {"status": "no_datasets"}
    method_names = None
    for methods in datasets.values():
        current = set(_hybrid_methods(methods))
        method_names = current if method_names is None else method_names.intersection(current)
    if not method_names:
        return {"status": "no_common_hybrid_candidates"}

    feasible = []
    scored = []
    for method in sorted(method_names):
        deltas = []
        f1_values = []
        latencies = []
        params = None
        ok = True
        for methods in datasets.values():
            baseline = _baseline_f1(methods, baseline_method)
            values = methods[method]
            delta = float(values["f1"]) - baseline
            deltas.append(delta)
            f1_values.append(float(values["f1"]))
            latencies.append(_latency(values))
            params = values.get("parameters", {})
            if delta < -max_f1_drop:
                ok = False
        record = {
            "method": method,
            "parameters": params or {},
            "mean_f1": sum(f1_values) / len(f1_values),
            "mean_latency_s": sum(latencies) / len(latencies),
            "min_f1_delta_vs_baseline": min(deltas),
        }
        scored.append(record)
        if ok:
            feasible.append(record)
    pool = feasible or scored
    selected = min(pool, key=lambda item: _candidate_key(item["method"], {"parameters": item["parameters"], "prefill_latency_s_mean": item["mean_latency_s"]}))
    selected = dict(selected)
    selected["status"] = "feasible_all_datasets" if feasible else "fallback_best_available"
    selected["feasible_count"] = len(feasible)
    selected["common_hybrid_candidate_count"] = len(scored)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/hybrid/summary.json")
    parser.add_argument("--output", default="results/hybrid/selected_config.json")
    parser.add_argument("--baseline-method", default="receiver_full")
    parser.add_argument("--max-f1-drop", type=float, default=0.0, help="absolute F1 drop allowed; default requires no drop")
    args = parser.parse_args()

    summary = json.loads(_resolve(args.summary).read_text(encoding="utf-8"))
    per_dataset = {
        dataset: _select_per_dataset(methods, args.baseline_method, args.max_f1_drop)
        for dataset, methods in summary.get("datasets", {}).items()
    }
    report = {
        "selection_rule_zh": "以receiver_full为质量下限，默认不允许F1下降；在满足质量约束的候选中优先选择估算重算工作量最低的DroidBlend配置。",
        "parameter_meaning_zh": {
            "recompute_layers": "任意连续全量重算层段[start,end]；第0层固定用于token选择并额外全量重算",
            "token_recompute_ratio": "不在全量重算层段、且不是第0层的其他层继续重算并回填KV的token比例",
            "recompute_work_ratio": "估算重算工作量比例，公式为 (full_layers + sparse_layers*token_ratio) / L",
        },
        "baseline_method": args.baseline_method,
        "max_f1_drop": args.max_f1_drop,
        "global_selection": _select_global(summary, args.baseline_method, args.max_f1_drop),
        "per_dataset_selection": per_dataset,
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_config": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


