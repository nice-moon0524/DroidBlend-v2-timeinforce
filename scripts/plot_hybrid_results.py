"""Plot DroidBlend hybrid summary files.

The input is ``results/hybrid/summary.json`` produced by
``experiments.run_hybrid_experiment``.  Figures are report-friendly: a
quality/latency scatter per dataset plus a recompute-span/token-ratio heatmap
for hybrid candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METHOD_STYLES = {
    "receiver_full": {"marker": "o", "color": "#1f77b4", "label": "Receiver full"},
    "full_kv_reuse": {"marker": "s", "color": "#ff7f0e", "label": "Full KV reuse"},
    "token_only": {"marker": "D", "color": "#9467bd", "label": "Token-only"},
    "hybrid": {"marker": "P", "color": "#2ca02c", "label": "DroidBlend"},
    "droidspeak": {"marker": "^", "color": "#d62728", "label": "DroidSpeak Pareto"},
}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _method_family(method: str) -> str:
    if method.startswith("hybrid_layers_") or method.startswith("hybrid_dense_"):
        return "hybrid"
    if method.startswith("droidspeak_layers_"):
        return "droidspeak"
    if method.startswith("token_only_ratio_"):
        return "token_only"
    return method


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _metric(values: dict[str, Any], key: str) -> float:
    if key in values:
        return float(values[key])
    aliases = {"prefill_latency_s_mean": "prefill_latency_s"}
    if key in aliases and aliases[key] in values:
        return float(values[aliases[key]])
    return 0.0


def _span_label(params: dict[str, Any]) -> str:
    span = params.get("recompute_layers")
    if isinstance(span, list) and len(span) == 2:
        return f"{int(span[0])}-{int(span[1])}"
    dense = params.get("dense_layer_count")
    if dense is not None:
        return f"0-{int(dense) - 1}"
    return "unknown"


def _span_sort_key(label: str) -> tuple[int, int]:
    try:
        start, end = label.split("-", 1)
        return int(start), int(end)
    except ValueError:
        return 9999, 9999


def _plot_dataset(dataset: str, methods: dict[str, Any], output_dir: Path) -> None:
    plt.figure(figsize=(6.2, 4.0))
    seen_labels: set[str] = set()
    for method, values in methods.items():
        family = _method_family(method)
        style = METHOD_STYLES.get(family, {"marker": "x", "color": "#555555", "label": family})
        label = style["label"] if style["label"] not in seen_labels else None
        seen_labels.add(style["label"])
        plt.scatter(
            _metric(values, "prefill_latency_s_mean"),
            _metric(values, "f1") * 100,
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=0.7,
            s=80 if family != "hybrid" else 58,
            alpha=0.92 if family != "hybrid" else 0.72,
            label=label,
        )
    plt.title(dataset)
    plt.xlabel("Prefill + first-token latency (s)")
    plt.ylabel("QA F1 (%)")
    plt.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        plt.savefig(output_dir / f"quality_latency_{_slug(dataset)}.{fmt}", dpi=200 if fmt == "png" else None)
    plt.close()


def _plot_hybrid_heatmap(dataset: str, methods: dict[str, Any], output_dir: Path) -> None:
    points = []
    for method, values in methods.items():
        if not (method.startswith("hybrid_layers_") or method.startswith("hybrid_dense_")):
            continue
        params = values.get("parameters", {})
        if "token_recompute_ratio" not in params:
            continue
        points.append((_span_label(params), float(params["token_recompute_ratio"]), float(values["f1"])))
    if not points:
        return
    span_values = sorted({item[0] for item in points}, key=_span_sort_key)
    ratio_values = sorted({item[1] for item in points})
    table = [[float("nan") for _ in span_values] for _ in ratio_values]
    for span, ratio, f1 in points:
        table[ratio_values.index(ratio)][span_values.index(span)] = f1 * 100

    plt.figure(figsize=(max(6.0, len(span_values) * 0.55), max(3.4, len(ratio_values) * 0.38)))
    image = plt.imshow(table, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(span_values)), span_values, rotation=35, ha="right")
    plt.yticks(range(len(ratio_values)), [f"{ratio:.2f}" for ratio in ratio_values])
    plt.xlabel("Full recompute span [start-end]")
    plt.ylabel("Token recompute ratio")
    plt.title(f"DroidBlend F1 heatmap: {dataset}")
    plt.colorbar(image, label="QA F1 (%)")
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        plt.savefig(output_dir / f"hybrid_f1_heatmap_{_slug(dataset)}.{fmt}", dpi=200 if fmt == "png" else None)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/hybrid/summary.json")
    parser.add_argument("--output-dir", default="results/hybrid/figures")
    args = parser.parse_args()

    summary = json.loads(_resolve(args.summary).read_text(encoding="utf-8"))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset, methods in summary.get("datasets", {}).items():
        _plot_dataset(dataset, methods, output_dir)
        _plot_hybrid_heatmap(dataset, methods, output_dir)
    print(json.dumps({"figures": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
