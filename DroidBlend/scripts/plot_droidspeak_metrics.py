#!/usr/bin/env python3
"""绘制 DroidSpeak 复现的论文式性能图。

本脚本只读取 DroidBlend 已落盘的 JSON 结果，不加载模型、不重跑推理。它按
论文的证据链依次绘制：接收端优势、单层 KV 敏感度、质量-预填充延迟折中，以及
保留集上的最终质量/本地 prefill 对比。

当前 DroidBlend 是单机单 GPU 复现，图中的 ``local prefill`` 包括缓存从主存
物化到 GPU 与 B 的重算层计算，不包含跨节点网络传输、排队、TTFT/TBT/QPS。
这些分布式系统指标需要后续 vLLM/多节点实验，不能由本脚本虚构。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _evaluation_payload(root: Path, explicit: str | None) -> tuple[dict[str, Any] | None, Path | None]:
    if explicit:
        path = Path(explicit)
        return _read(path), path
    directory = root / "evaluation"
    candidates = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not candidates:
        return None, None
    return _read(candidates[-1]), candidates[-1]


def _setup_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("缺少 matplotlib；请在实验环境安装 matplotlib 后重试。") from exc
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(figure: Any, path: Path, dpi: int) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)


def plot_baseline(plt: Any, baseline: dict[str, Any], report_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    sender = baseline["sender_A_native"]
    receiver = baseline["receiver_B_native"]
    sender_f1 = float(sender["quality"]["qa_f1"]) * 100
    receiver_f1 = float(receiver["quality"]["qa_f1"]) * 100
    sender_ms = float(sender["mean_prefill_ms"])
    receiver_ms = float(receiver["mean_prefill_ms"])
    figure, (quality_ax, latency_ax) = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = ["Sender A\nBase", "Receiver B\nInstruct"]
    colors = ["#7F8C8D", "#2E86AB"]
    bars = quality_ax.bar(labels, [sender_f1, receiver_f1], color=colors)
    quality_ax.bar_label(bars, fmt="%.2f", padding=3)
    quality_ax.set_ylabel("QA-F1 (%)")
    quality_ax.set_title("Receiver task-quality gate")
    bars = latency_ax.bar(labels, [sender_ms, receiver_ms], color=colors)
    latency_ax.bar_label(bars, fmt="%.1f ms", padding=3)
    latency_ax.set_ylabel("Mean native prefix prefill (ms)")
    latency_ax.set_title("Native prefill reference")
    output = report_dir / "01_sender_receiver_baseline.png"
    return _save(figure, output, dpi), {
        "sender_qa_f1": sender_f1,
        "receiver_qa_f1": receiver_f1,
        "receiver_minus_sender_f1_points": receiver_f1 - sender_f1,
        "direction_gate_passed": bool(baseline.get("droid_direction_gate_passed", False)),
    }


def plot_sensitivity(plt: Any, sensitivity: dict[str, Any], report_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    native = float(sensitivity["native"]["qa_f1"])
    rows = sensitivity["layers"]
    layers = [int(row["layer"]) for row in rows]
    values = [float(row["scores"]["qa_f1"]) * 100 for row in rows]
    relative_drops = [max(0.0, float(row["qa_f1_drop_absolute"]) / max(abs(native), 1e-12)) for row in rows]
    critical = [layer for layer, drop in zip(layers, relative_drops, strict=True) if drop > 0.10]
    colors = ["#C44E52" if layer in critical else "#4C78A8" for layer in layers]
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.bar(layers, values, color=colors, width=0.8, label="B with one A K/V layer")
    axis.axhline(native * 100, color="#222222", linestyle="--", linewidth=1.2, label="Native B")
    axis.set(xlabel="Layer replaced by sender A K/V", ylabel="QA-F1 (%)", title="Layer-wise sensitivity to cross-model K/V reuse")
    axis.set_xticks(layers)
    axis.legend(loc="best")
    axis.text(0.99, 0.03, f"critical: {len(critical)}/{len(layers)} (relative F1 drop > 10%)", transform=axis.transAxes, ha="right", va="bottom")
    output = report_dir / "02_layer_sensitivity.png"
    return _save(figure, output, dpi), {"critical_layers": critical, "critical_layer_fraction": len(critical) / max(1, len(layers))}


def plot_direct_reuse(plt: Any, baseline: dict[str, Any], direct: dict[str, Any], report_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    """Paper Figure-3 style comparison: A, direct full-KV reuse, and native B."""
    sender = baseline["sender_A_native"]["quality"]
    receiver = direct["native_receiver"]
    direct_quality = direct["direct_sender_kv"]["quality"]
    labels = ["Sender A\nnative", "Receiver B\ndirect all A-KV", "Receiver B\nnative"]
    colors = ["#7F8C8D", "#F28E2B", "#2E86AB"]
    figure, (f1_ax, em_ax) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    f1_values = [float(sender["qa_f1"]) * 100, float(direct_quality["qa_f1"]) * 100, float(receiver["qa_f1"]) * 100]
    em_values = [float(sender["exact_match"]) * 100, float(direct_quality["exact_match"]) * 100, float(receiver["exact_match"]) * 100]
    for axis, values, metric in ((f1_ax, f1_values, "QA-F1 (%)"), (em_ax, em_values, "Exact match (%)")):
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", labelrotation=8)
    f1_ax.set_title("Full sender-KV reuse diagnostic")
    em_ax.set_title("Exact-match comparison")
    output = report_dir / "03_full_kv_reuse_comparison.png"
    return _save(figure, output, dpi), {
        "sender_qa_f1": f1_values[0],
        "direct_all_sender_kv_qa_f1": f1_values[1],
        "native_receiver_qa_f1": f1_values[2],
        "native_receiver_relative_loss_with_direct_reuse": float(direct["relative_qa_f1_loss"]),
    }


def plot_targeted_stability(plt: Any, stability: dict[str, Any], report_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    """Plot paired bootstrap intervals for the targeted layer-swap diagnostic."""
    rows = sorted(stability["layers"], key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in rows]
    means = [float(row["mean_native_minus_swap_qa_f1"]) * 100 for row in rows]
    lows = [float(row["bootstrap_95pct_ci_native_minus_swap_qa_f1"][0]) * 100 for row in rows]
    highs = [float(row["bootstrap_95pct_ci_native_minus_swap_qa_f1"][1]) * 100 for row in rows]
    lower_errors = [value - low for value, low in zip(means, lows, strict=True)]
    upper_errors = [high - value for value, high in zip(means, highs, strict=True)]
    stable_harm = [low > 0 for low in lows]
    colors = ["#C44E52" if harmful else "#4C78A8" for harmful in stable_harm]

    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    y = list(range(len(rows)))
    axis.errorbar(
        means,
        y,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#555555",
        elinewidth=1.4,
        capsize=4,
        zorder=1,
    )
    axis.scatter(means, y, c=colors, s=58, zorder=2)
    axis.axvline(0, color="#222222", linestyle="--", linewidth=1.1)
    axis.set_yticks(y, [f"Layer {layer}" for layer in layers])
    axis.invert_yaxis()
    axis.set_xlabel("Native B QA-F1 − single-layer A-K/V swap (points)")
    axis.set_title("Targeted layer-swap stability (paired bootstrap 95% CI)")
    axis.text(
        0.99,
        0.03,
        "Right of zero: swap lowers quality; CI crossing zero: inconclusive",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    output = report_dir / "04_targeted_sensitivity_stability.png"
    return _save(figure, output, dpi), {
        "layers": layers,
        "stable_harmful_layers": [layer for layer, harmful in zip(layers, stable_harm, strict=True) if harmful],
    }


def plot_pareto(plt: Any, profiles: dict[str, Any], selected: dict[str, Any] | None, report_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    points = profiles["points"]
    native_quality = float(profiles["native_quality"])
    x = [float(point["hybrid_prefill_ms"]) for point in points]
    y = [float(point["quality"]) * 100 for point in points]
    sizes = [int(point["group"]["end"]) - int(point["group"]["start"]) for point in points]
    native_ms = float(points[0]["native_prefill_ms"])
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    scatter = axis.scatter(x, y, c=sizes, cmap="viridis", s=42, alpha=0.88, label="DroidSpeak layer groups")
    figure.colorbar(scatter, ax=axis, label="Recomputed-layer count")
    axis.scatter([native_ms], [native_quality * 100], marker="*", s=180, color="#222222", label="Native B full prefill", zorder=3)
    selected_id = None
    if selected and selected.get("selected"):
        chosen = selected["selected"]
        selected_id = chosen["group"]["identifier"]
        axis.scatter([float(chosen["hybrid_prefill_ms"])], [float(chosen["quality"]) * 100], marker="D", s=78, color="#C44E52", label=f"Selected: {selected_id}", zorder=4)
    axis.set(xlabel="Local hybrid prefill latency (ms)", ylabel="QA-F1 (%)", title="Quality - prefill latency trade-off")
    axis.legend(loc="best")
    output = report_dir / "05_quality_prefill_pareto.png"
    fastest = min(points, key=lambda point: float(point["hybrid_prefill_ms"]))
    return _save(figure, output, dpi), {"profile_points": len(points), "selected_group": selected_id, "fastest_group": fastest["group"]["identifier"]}


def plot_heldout(plt: Any, evaluation: dict[str, Any], report_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    methods = ["Native B", "Direct A KV reuse", "DroidSpeak"]
    quality = [
        float(evaluation["native"]["qa_f1"]) * 100,
        float(evaluation["raw_sender_kv"]["qa_f1"]) * 100,
        float(evaluation["droidspeak_hybrid"]["qa_f1"]) * 100,
    ]
    local = evaluation.get("local_prefill_performance", {})
    rows = evaluation.get("per_example") or []
    native_ms = local.get("native_receiver_prefill_ms") or _mean_present(rows, "native_prefix_ms")
    hybrid_ms = local.get("droidspeak_hybrid_prefill_ms") or _mean_present(rows, "hybrid_selected_blocks_ms")
    figure, (quality_ax, latency_ax) = plt.subplots(1, 2, figsize=(10, 3.9))
    colors = ["#2E86AB", "#B0B0B0", "#59A14F"]
    bars = quality_ax.bar(methods, quality, color=colors)
    quality_ax.bar_label(bars, fmt="%.2f", padding=3)
    quality_ax.set_ylabel("Held-out QA-F1 (%)")
    quality_ax.set_title("Held-out generation quality")
    quality_ax.tick_params(axis="x", labelrotation=12)
    result: dict[str, Any] = {"heldout_native_f1": quality[0], "heldout_direct_reuse_f1": quality[1], "heldout_droidspeak_f1": quality[2]}
    if native_ms is not None and hybrid_ms is not None:
        bars = latency_ax.bar(["Native B\nfull prefill", "DroidSpeak\nlocal hybrid prefill"], [native_ms, hybrid_ms], color=["#2E86AB", "#59A14F"])
        latency_ax.bar_label(bars, fmt="%.1f ms", padding=3)
        latency_ax.set_ylabel("Latency (ms)")
        speedup = native_ms / hybrid_ms if hybrid_ms > 0 else None
        latency_ax.set_title(f"Local prefill speedup: {speedup:.2f}x" if speedup else "Local prefill")
        result["heldout_local_prefill_speedup"] = speedup
    else:
        latency_ax.text(0.5, 0.5, "No timing fields in evaluation JSON", ha="center", va="center")
        latency_ax.set_axis_off()
    output = report_dir / "06_heldout_quality_and_prefill.png"
    return _save(figure, output, dpi), result


def main() -> int:
    parser = argparse.ArgumentParser(description="从 DroidBlend JSON 结果绘制 DroidSpeak 性能图")
    parser.add_argument("--output-dir", required=True, help="实验 output_dir，例如 outputs/llama31_8b_base_to_instruct")
    parser.add_argument("--evaluation", default=None, help="可选：指定某个保留集 evaluation JSON")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    root = Path(args.output_dir)
    report_dir = root / "paper_figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()
    rendered: dict[str, str] = {}
    metrics: dict[str, Any] = {"measurement_scope": "single-GPU local reproduction; no distributed network/queueing metrics"}

    baseline = _read(root / "baseline" / "native_pair_calibration.json")
    if baseline:
        rendered["baseline"] , metrics["baseline"] = plot_baseline(plt, baseline, report_dir, args.dpi)
    sensitivity = _read(root / "sensitivity" / "single_layer_kv_swap.json")
    if sensitivity:
        rendered["sensitivity"], metrics["sensitivity"] = plot_sensitivity(plt, sensitivity, report_dir, args.dpi)
    direct_reuse = _read(root / "diagnostics" / "direct_reuse_calibration.json")
    if baseline and direct_reuse:
        rendered["direct_reuse"], metrics["direct_reuse"] = plot_direct_reuse(plt, baseline, direct_reuse, report_dir, args.dpi)
    stability = _read(root / "diagnostics" / "targeted_sensitivity_stability.json")
    if stability:
        rendered["targeted_stability"], metrics["targeted_stability"] = plot_targeted_stability(plt, stability, report_dir, args.dpi)
    profiles = _read(root / "profiling" / "points.json")
    if profiles:
        rendered["pareto"], metrics["pareto"] = plot_pareto(plt, profiles, _read(root / "profiling" / "selected.json"), report_dir, args.dpi)
    evaluation, evaluation_path = _evaluation_payload(root, args.evaluation)
    if evaluation:
        rendered["heldout"], metrics["heldout"] = plot_heldout(plt, evaluation, report_dir, args.dpi)
        metrics["heldout"]["source"] = str(evaluation_path)
    if not rendered:
        raise SystemExit(f"在 {root} 下没有找到可绘制的 DroidBlend 结果 JSON。")
    summary = report_dir / "metrics_summary.json"
    with summary.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"figures": rendered, "summary": str(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
