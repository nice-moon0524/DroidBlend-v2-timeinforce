from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json_write
from .config import ExperimentConfig
from .errors import ArtifactError
from .pareto import ProfilePoint, pareto_frontier, select_fastest_within_budget
from .schedules import LayerGroup


def read_profile_points(path: Path) -> list[ProfilePoint]:
    if not path.is_file():
        raise ArtifactError(f"Profile results are missing: {path}. Run profile first.")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    points = []
    for item in raw.get("points", []):
        group = item["group"]
        points.append(
            ProfilePoint(
                group=LayerGroup(start=int(group["start"]), end=int(group["end"])),
                quality=float(item["quality"]),
                native_quality=float(item["native_quality"]),
                quality_metric=str(item["quality_metric"]),
                hybrid_prefill_ms=float(item["hybrid_prefill_ms"]),
                native_prefill_ms=float(item["native_prefill_ms"]),
                transfer_bytes=int(item["transfer_bytes"]),
                cache_error=float(item["cache_error"]) if item.get("cache_error") is not None else None,
                samples=int(item["samples"]),
            )
        )
    if not points:
        raise ArtifactError(f"No profile points in {path}")
    return points


def select_profile(config: ExperimentConfig) -> dict[str, Any]:
    points_path = config.output_dir / "profiling" / "points.json"
    points = read_profile_points(points_path)
    chosen = select_fastest_within_budget(points, config.profiling.quality_tolerance_relative)
    frontier = pareto_frontier(points, config.profiling.quality_tolerance_relative)
    output = config.output_dir / "profiling" / "selected.json"
    atomic_json_write(
        output,
        {
            "selection_rule": "fastest hybrid prefill within relative quality-loss budget",
            "quality_tolerance_relative": config.profiling.quality_tolerance_relative,
            "selected": chosen.to_dict(),
            "pareto_frontier": [point.to_dict() for point in frontier],
        },
    )
    return {"selected_path": str(output), "selected_group": chosen.group.identifier, "frontier_points": len(frontier)}


def resolve_selected_group(config: ExperimentConfig, explicit: str | None = None) -> LayerGroup:
    value = explicit or config.evaluation.selected_profile
    if value and value.startswith("layers_"):
        parts = value.split("_")
        if len(parts) != 3:
            raise ArtifactError(f"Invalid selected layer identifier: {value}")
        return LayerGroup(int(parts[1]), int(parts[2]) + 1)
    selected_path = config.output_dir / "profiling" / "selected.json"
    if not selected_path.is_file():
        raise ArtifactError("No selected profile. Run select-pareto or set evaluation.selected_profile=layers_XXX_YYY")
    with selected_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)["selected"]["group"]
    return LayerGroup(int(selected["start"]), int(selected["end"]))
