"""Create publication-style figures for the token-anchor K/KV ablation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


GROUPS = ["layer0-k", "layer0-kv", "start-k", "start-kv"]
GROUP_LABELS = {
    "layer0-k": "Layer 0 / K",
    "layer0-kv": "Layer 0 / K+V",
    "start-k": "Span start / K",
    "start-kv": "Span start / K+V",
}
COLORS = {"layer0-k": "#0072B2", "layer0-kv": "#E69F00", "start-k": "#009E73", "start-kv": "#CC79A7"}
MARKERS = {"layer0-k": "o", "layer0-kv": "s", "start-k": "^", "start-kv": "D"}


def _group(params: dict[str, Any]) -> str:
    return f"{params['anchor_mode']}-{params['selection_score_mode']}"


def _load_summary(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for method, values in raw["methods"].items():
        params = values["parameters"]
        start, end = params["recompute_layers"]
        rows.append({
            "method": method,
            "group": _group(params),
            "span": f"{start}-{end}",
            "start": int(start),
            "end": int(end),
            "full_layers": int(params["full_recompute_layer_count"]),
            "f1": float(values["f1"]),
            "latency_s": float(values["prefill_latency_s_mean"]),
            "delta_f1": float(values["f1_vs_droidspeak"]),
            "delta_latency_s": float(values["latency_vs_droidspeak"]),
            "baseline_f1": float(params["droidspeak_baseline_f1"]),
            "baseline_latency_s": float(params["droidspeak_baseline_prefill_latency_s_mean"]),
            "work_ratio": float(params["recompute_work_ratio"]),
        })
    frame = pd.DataFrame(rows)
    frame["delta_latency_ms"] = frame["delta_latency_s"] * 1_000
    frame["latency_ms"] = frame["latency_s"] * 1_000
    frame["baseline_latency_ms"] = frame["baseline_latency_s"] * 1_000
    frame["passes_droidspeak_f1"] = frame["delta_f1"] >= -1e-12
    return raw, frame.sort_values(["full_layers", "start", "end", "group"]).reset_index(drop=True)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(figure, directory: Path, stem: str) -> None:
    figure.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _quality_by_span(frame: pd.DataFrame, directory: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    spans = frame[["span", "full_layers", "start", "end"]].drop_duplicates().sort_values(["full_layers", "start", "end"])
    x = np.arange(len(spans))
    baseline = frame.drop_duplicates("span").set_index("span").loc[spans["span"]]
    labels = [f"{span}\n({layers})" for span, layers in zip(spans["span"], spans["full_layers"])]
    for axis, group in zip(axes.flat, GROUPS):
        values = frame[frame["group"] == group].set_index("span").loc[spans["span"]]
        axis.plot(x, baseline["baseline_f1"], color="#4D4D4D", linestyle="--", marker="o", markersize=3.5, linewidth=1.2, label="DroidSpeak")
        axis.plot(x, values["f1"], color=COLORS[group], marker=MARKERS[group], markersize=5, linewidth=1.8, label=GROUP_LABELS[group])
        failing = values[~values["passes_droidspeak_f1"]]
        if not failing.empty:
            positions = [int(np.where(spans["span"].to_numpy() == span)[0][0]) for span in failing.index]
            axis.scatter(positions, failing["f1"], facecolors="none", edgecolors="#B2182B", linewidths=1.4, s=68, zorder=4)
        axis.set_title(GROUP_LABELS[group])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_ylim(0.2, 0.9)
        axis.legend(loc="lower right", frameon=False)
    for axis in axes[-1, :]:
        axis.set_xticks(x, labels, rotation=0)
        axis.set_xlabel("DroidSpeak dense span (full layers)")
    for axis in axes[:, 0]:
        axis.set_ylabel("HotpotQA F1")
    figure.suptitle("Quality trajectory across locked DroidSpeak Pareto spans", y=1.02, fontsize=14)
    _save(figure, directory, "fig1_quality_by_span")


def _quality_latency(frame: pd.DataFrame, receiver_full_f1: float, directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.1), constrained_layout=True)
    baseline = frame.drop_duplicates("span")
    axis.scatter(baseline["baseline_latency_ms"], baseline["baseline_f1"], color="#4D4D4D", marker="x", s=38, linewidths=1.3, label="DroidSpeak baseline", zorder=3)
    for group in GROUPS:
        subset = frame[frame["group"] == group]
        axis.scatter(subset["latency_ms"], subset["f1"], color=COLORS[group], marker=MARKERS[group], s=55, alpha=0.88, label=GROUP_LABELS[group], zorder=4)
    axis.axhline(receiver_full_f1, color="#B2182B", linestyle="--", linewidth=1.1, label=f"Receiver full F1 = {receiver_full_f1:.3f}")
    axis.set_xlabel("Mean prefill latency (ms)")
    axis.set_ylabel("HotpotQA F1")
    axis.set_title("Quality-latency landscape: 16 spans x 4 token-selection strategies")
    axis.grid(color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    _save(figure, directory, "fig2_quality_latency_landscape")


def _delta_quadrant(frame: pd.DataFrame, directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.1), constrained_layout=True)
    x_min, x_max = frame["delta_latency_ms"].min(), frame["delta_latency_ms"].max()
    y_min, y_max = frame["delta_f1"].min(), frame["delta_f1"].max()
    x_pad = max(1.5, (x_max - x_min) * 0.08)
    y_pad = max(0.015, (y_max - y_min) * 0.10)
    axis.axvspan(x_min - x_pad, 0, color="#D9F0D3", alpha=0.55, zorder=0)
    axis.axvspan(0, x_max + x_pad, color="#FEE0D2", alpha=0.35, zorder=0)
    axis.axhline(0, color="#4D4D4D", linewidth=1.0)
    axis.axvline(0, color="#4D4D4D", linewidth=1.0)
    for group in GROUPS:
        subset = frame[frame["group"] == group]
        axis.scatter(subset["delta_latency_ms"], subset["delta_f1"], color=COLORS[group], marker=MARKERS[group], s=58, alpha=0.9, label=GROUP_LABELS[group])
    axis.annotate("Target quadrant\n(faster, no F1 drop)", xy=(-0.02, 0.98), xycoords="axes fraction", ha="right", va="top", color="#238443", fontsize=10)
    axis.annotate("Slower than\nDroidSpeak", xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom", color="#B2182B", fontsize=10)
    axis.set_xlim(x_min - x_pad, x_max + x_pad)
    axis.set_ylim(y_min - y_pad, y_max + y_pad)
    axis.set_xlabel("Latency change vs same-span DroidSpeak (ms)")
    axis.set_ylabel("F1 change vs same-span DroidSpeak")
    axis.set_title("Primary decision plot: does token correction improve the same span?")
    axis.grid(color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    _save(figure, directory, "fig3_delta_vs_droidspeak")


def _selection_overlap(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    methods = {(row.span, row.group): row.method for row in frame.itertuples()}
    overlaps: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)

            for span in frame["span"].unique():
                for anchor in ("layer0", "start"):
                    k_name = methods[(span, f"{anchor}-k")]
                    kv_name = methods[(span, f"{anchor}-kv")]
                    k_indices = set(sample["methods"][k_name]["selected_indices"])
                    kv_indices = set(sample["methods"][kv_name]["selected_indices"])
                    overlap = len(k_indices & kv_indices) / len(k_indices | kv_indices) if k_indices or kv_indices else 1.0
                    overlaps.append({"span": span, "anchor": anchor, "jaccard": overlap})
    return pd.DataFrame(overlaps).groupby(["anchor", "span"], as_index=False).agg(mean_jaccard=("jaccard", "mean"), std_jaccard=("jaccard", "std"))


def _overlap_heatmap(overlaps: pd.DataFrame, frame: pd.DataFrame, directory: Path) -> None:
    spans = frame[["span", "full_layers", "start", "end"]].drop_duplicates().sort_values(["full_layers", "start", "end"])
    figure, axes = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True, constrained_layout=True)
    for axis, anchor in zip(axes, ("layer0", "start")):
        values = overlaps[overlaps["anchor"] == anchor].set_index("span").loc[spans["span"], "mean_jaccard"].to_numpy()[None, :]
        image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for index, value in enumerate(values[0]):
            axis.text(index, 0, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=9)
        axis.set_yticks([0], ["Layer 0" if anchor == "layer0" else "Span start"])
        axis.set_title(f"K vs K+V selected-token Jaccard overlap: {GROUP_LABELS[anchor + '-k']}")
    axes[-1].set_xticks(np.arange(len(spans)), [f"{span}\n({layers})" for span, layers in zip(spans["span"], spans["full_layers"])])
    axes[-1].set_xlabel("DroidSpeak dense span (full layers)")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Mean Jaccard overlap across 50 prompts")
    _save(figure, directory, "fig4_k_vs_kv_token_overlap")


def _report(raw: dict[str, Any], frame: pd.DataFrame, overlaps: pd.DataFrame, path: Path) -> None:
    feasible = frame[frame["passes_droidspeak_f1"]]
    faster_feasible = feasible[feasible["delta_latency_ms"] < 0]
    best_quality = frame.sort_values(["f1", "latency_ms"], ascending=[False, True]).iloc[0]
    best_feasible_latency = feasible.sort_values("latency_ms").iloc[0] if not feasible.empty else None
    group_stats = frame.groupby("group", sort=False).agg(
        mean_f1=("f1", "mean"), mean_latency_ms=("latency_ms", "mean"), mean_delta_f1=("delta_f1", "mean"),
        mean_delta_latency_ms=("delta_latency_ms", "mean"), pass_count=("passes_droidspeak_f1", "sum"),
    ).reindex(GROUPS)
    overlap_lines = []
    for anchor in ("layer0", "start"):
        subset = overlaps[overlaps["anchor"] == anchor]
        overlap_lines.append(f"- `{anchor}`: mean Jaccard `{subset.mean_jaccard.mean():.3f}` (range `{subset.mean_jaccard.min():.3f}`-{subset.mean_jaccard.max():.3f}`)")
    lines = [
        "# Token Anchor K/KV 消融结果分析",
        "",
        "## 实验范围",
        f"- 数据集：`{raw['dataset']}`，{raw['samples']} 条样本；16 个锁定 DroidSpeak Pareto contiguous spans；每个 span 的 4 个策略均选择 20% 有效 token。",
        "- 延迟口径：仅 hybrid prefill 路径，未包含 sender cache 提取、decode 和 receiver full prefill。",
        "",
        "## 结论",
        f"- 通过同 span DroidSpeak F1 门槛的候选：`{len(feasible)}/64`。其中同时更快的候选：`{len(faster_feasible)}/64`。",
        "- 因此，本轮 20% token correction 没有进入目标象限（相对同 span DroidSpeak：F1 不下降且 latency 更低）。主要问题不是 F1，而是 token selection、span 外稀疏层重算和 KV merge 的额外成本超过了速度收益。",
        f"- 最高绝对 F1：`{best_quality['group']}` / span `{best_quality['span']}`，F1 `{best_quality['f1']:.4f}`，latency `{best_quality['latency_ms']:.2f}` ms。",
    ]
    if best_feasible_latency is not None:
        lines.append(f"- 通过 DroidSpeak F1 门槛的最低 latency 候选：`{best_feasible_latency['group']}` / span `{best_feasible_latency['span']}`，F1 `{best_feasible_latency['f1']:.4f}`，latency `{best_feasible_latency['latency_ms']:.2f}` ms，仍比同 span DroidSpeak 慢 `{best_feasible_latency['delta_latency_ms']:.2f}` ms。")
    lines.extend(["", "## 四组总体均值", "", "| 策略 | mean F1 | mean latency (ms) | mean ΔF1 vs DroidSpeak | mean Δlatency (ms) | F1 pass / 16 |", "|---|---:|---:|---:|---:|---:|"])
    for group, row in group_stats.iterrows():
        lines.append(f"| {GROUP_LABELS[group]} | {row.mean_f1:.4f} | {row.mean_latency_ms:.2f} | {row.mean_delta_f1:+.4f} | {row.mean_delta_latency_ms:+.2f} | {int(row.pass_count)}/16 |")
    lines.extend(["", "## K 与 K+V 选 token 的差异", "", *overlap_lines, "- Jaccard 越低，说明 K+V 越明显地改变了被选择的 token 集合。结合图 1 和图 3，应以同一 span 下的质量增益是否能抵消 latency 代价来判断是否保留 V。", "", "## 图表解读", "", "- `fig1_quality_by_span`：每一个面板只改变一个策略，黑色虚线是该 span 的 DroidSpeak 基线；红色空心圈表示未达到同 span F1。", "- `fig2_quality_latency_landscape`：展示所有候选和 DroidSpeak 的绝对质量-时延位置；红虚线是 receiver full 的 F1，仅作质量参照。", "- `fig3_delta_vs_droidspeak`：主决策图。左上区域才满足本实验目标；本轮候选应重点观察其为何全部位于右侧。", "- `fig4_k_vs_kv_token_overlap`：衡量 K+V 是否真正改变 token selection，而非只增加投影开销。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot token-anchor ablation results")
    parser.add_argument("--results-dir", default="results/token_anchor_ablation/hotpotqa_50_ratio_0p20")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, frame = _load_summary(results_dir / "summary.json")
    if len(frame) != 64:
        raise ValueError(f"expected 64 methods (16 spans x 4 groups), found {len(frame)}")
    _style()
    _quality_by_span(frame, output_dir)
    first_method = next(iter(raw["methods"].values()))
    receiver_full_f1 = float(first_method["f1"]) - float(first_method["f1_vs_receiver_full"])
    _quality_latency(frame, receiver_full_f1, output_dir)
    _delta_quadrant(frame, output_dir)
    overlaps = _selection_overlap(results_dir / "per_example.jsonl", frame)
    _overlap_heatmap(overlaps, frame, output_dir)
    frame.sort_values(["passes_droidspeak_f1", "latency_ms", "f1"], ascending=[False, True, False]).to_csv(output_dir / "method_comparison.csv", index=False)
    overlaps.to_csv(output_dir / "token_overlap.csv", index=False)
    _report(raw, frame, overlaps, output_dir / "analysis_report_zh.md")
    print(json.dumps({"figures": str(output_dir), "report": str(output_dir / "analysis_report_zh.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
