from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import ConfigurationError


@dataclass(frozen=True, order=True)
class LayerGroup:
    """A half-open continuous receiver recomputation interval [start, end)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ConfigurationError(f"Invalid recomputation layer group [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def identifier(self) -> str:
        return f"layers_{self.start:03d}_{self.end - 1:03d}"

    def layers(self) -> list[int]:
        return list(range(self.start, self.end))


def _normalize_values(value: str | Iterable[int], upper: int, label: str) -> list[int]:
    if value == "all":
        return list(range(upper))
    values = sorted({int(item) for item in value})
    if any(item < 0 or item >= upper for item in values):
        raise ConfigurationError(f"{label} must be within [0, {upper - 1}]")
    return values


def enumerate_layer_groups(
    num_layers: int,
    recompute_lengths: str | Iterable[int],
    min_recompute_layers: int = 1,
    max_recompute_layers: int | None = None,
    start_layers: str | Iterable[int] = "all",
) -> list[LayerGroup]:
    """Enumerate every legal continuous group required for DroidSpeak profiling."""
    if num_layers < 1:
        raise ConfigurationError("num_layers must be positive")
    max_length = num_layers if max_recompute_layers is None else min(max_recompute_layers, num_layers)
    if min_recompute_layers < 1 or min_recompute_layers > max_length:
        raise ConfigurationError("Invalid min/max recomputation length")

    if recompute_lengths == "all":
        lengths = list(range(min_recompute_layers, max_length + 1))
    else:
        lengths = sorted({int(item) for item in recompute_lengths})
        if any(item < min_recompute_layers or item > max_length for item in lengths):
            raise ConfigurationError("recompute_lengths contains an invalid value")

    allowed_starts = set(_normalize_values(start_layers, num_layers, "start_layers"))
    groups = [
        LayerGroup(start=start, end=start + length)
        for length in lengths
        for start in range(0, num_layers - length + 1)
        if start in allowed_starts
    ]
    if not groups:
        raise ConfigurationError("No legal layer groups remain after applying schedule constraints")
    return groups


def transition_layers(groups: Iterable[LayerGroup]) -> list[int]:
    """Return E-cache layers that must be captured for a set of candidate groups."""
    return sorted({group.start for group in groups})
