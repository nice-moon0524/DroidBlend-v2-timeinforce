from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .errors import ArtifactError


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"Required result file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_reports(config: ExperimentConfig) -> dict[str, str]:
    """Produce the two diagnostic figures used to interpret a Droid run."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("Report rendering needs matplotlib; install the [analysis] extra") from exc

    sensitivity = _read(config.output_dir / "sensitivity" / "single_layer_kv_swap.json")
    profiles = _read(config.output_dir / "profiling" / "points.json")
    report_dir = config.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    layer_rows = sensitivity["layers"]
    xs = [row["layer"] for row in layer_rows]
    drops = [row["qa_f1_drop_absolute"] for row in layer_rows]
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.bar(xs, drops, color="#C44E52")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Layer whose native B K/V is replaced by A K/V", ylabel="Absolute QA-F1 drop", title="Single-layer cache-swap sensitivity")
    axis.set_xticks(xs)
    figure.tight_layout()
    sensitivity_png = report_dir / "sensitivity.png"
    figure.savefig(sensitivity_png, dpi=200)
    plt.close(figure)

    points = profiles["points"]
    native_quality = float(profiles["native_quality"])
    quality_loss = [max(0.0, (native_quality - float(row["quality"])) / max(abs(native_quality), 1e-12)) for row in points]
    speedup = [float(row["prefill_speedup"]) for row in points]
    lengths = [int(row["group"]["end"]) - int(row["group"]["start"]) for row in points]
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(quality_loss, speedup, c=lengths, cmap="viridis", alpha=0.85)
    figure.colorbar(scatter, ax=axis, label="Recomputed-layer count")
    axis.axvline(config.profiling.quality_tolerance_relative, color="#C44E52", linestyle="--", label="quality-loss budget")
    axis.set(xlabel="Relative QA-F1 loss vs native B", ylabel="Native B prefill / hybrid prefill", title="Droid continuous-layer-group profile")
    axis.legend()
    figure.tight_layout()
    pareto_png = report_dir / "pareto.png"
    figure.savefig(pareto_png, dpi=200)
    plt.close(figure)
    return {"sensitivity_plot": str(sensitivity_png), "pareto_plot": str(pareto_png)}
