"""Render compact plots from DroidBlend JSON result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_STYLES = {
    "full_prefill": {"marker": "o", "color": "#1f77b4", "label": "Full prefill"},
    "full_kv_reuse": {"marker": "s", "color": "#ff7f0e", "label": "Full KV reuse"},
    "droidspeak": {"marker": "^", "color": "#2ca02c", "label": "DroidSpeak"},
    "cacheblend_reference": {"marker": "D", "color": "#d62728", "label": "CacheBlend ref."},
    "droidblend_reference": {"marker": "P", "color": "#9467bd", "label": "DroidBlend ref."},
}


def _slugify(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _plot_quality_for_dataset(output: Path, dataset: str, methods: dict[str, dict[str, float]]) -> None:
    plt.figure(figsize=(5.2, 3.4))
    for method, values in methods.items():
        style = METHOD_STYLES.get(method, {"marker": "X", "color": "#7f7f7f", "label": method})
        plt.scatter(
            values["prefill_latency_s"],
            values["f1"] * 100,
            marker=style["marker"],
            s=76,
            color=style["color"],
            edgecolors="white",
            linewidths=0.8,
            label=style["label"],
        )
    plt.title(dataset)
    plt.xlabel("Prefill latency (s)")
    plt.ylabel("Token F1 (%)")
    plt.legend(fontsize=7, frameon=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidblend", default="results/droidblend_quality_latency.json")
    parser.add_argument("--token-selection", default="results/droidblend_token_selection.json")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    quality = json.loads(_resolve(args.droidblend).read_text(encoding="utf-8"))
    for dataset, methods in quality["methods"].items():
        slug = _slugify(dataset)
        _plot_quality_for_dataset(output / f"droidblend_quality_latency_{slug}.pdf", dataset, methods)
        _plot_quality_for_dataset(output / f"droidblend_quality_latency_{slug}.png", dataset, methods)

    token_selection = json.loads(_resolve(args.token_selection).read_text(encoding="utf-8"))
    datasets = list(token_selection["datasets"])
    selected = [token_selection["datasets"][name]["average_selected_count"] for name in datasets]
    candidates = [token_selection["datasets"][name]["average_candidate_count"] for name in datasets]
    fractions = [(s / c * 100) if c else 0 for s, c in zip(selected, candidates, strict=True)]
    plt.figure(figsize=(5, 3.2))
    plt.bar(datasets, fractions, color="#009E73")
    plt.ylabel("Selected HKVD tokens (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output / "droidblend_token_selection.pdf")
    plt.close()


if __name__ == "__main__":
    main()
