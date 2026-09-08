from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .schedules import LayerGroup


@dataclass(frozen=True)
class ProfilePoint:
    group: LayerGroup
    quality: float
    native_quality: float
    quality_metric: str
    hybrid_prefill_ms: float
    native_prefill_ms: float
    transfer_bytes: int
    cache_error: float | None
    samples: int

    @property
    def quality_loss_relative(self) -> float:
        if self.native_quality == 0:
            return 0.0
        return max(0.0, (self.native_quality - self.quality) / abs(self.native_quality))

    @property
    def prefill_speedup(self) -> float:
        return self.native_prefill_ms / self.hybrid_prefill_ms if self.hybrid_prefill_ms > 0 else 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["group"] = {"start": self.group.start, "end": self.group.end, "identifier": self.group.identifier}
        payload["quality_loss_relative"] = self.quality_loss_relative
        payload["prefill_speedup"] = self.prefill_speedup
        return payload


def pareto_frontier(points: Iterable[ProfilePoint], tolerance_relative: float) -> list[ProfilePoint]:
    """Return latency-optimal points that meet a quality budget.

    A point is dominated if another feasible point has no worse latency and no lower
    quality, with at least one strict improvement.
    """
    feasible = [point for point in points if point.quality_loss_relative <= tolerance_relative]
    frontier: list[ProfilePoint] = []
    for point in sorted(feasible, key=lambda item: (item.hybrid_prefill_ms, -item.quality)):
        dominated = any(
            candidate.hybrid_prefill_ms <= point.hybrid_prefill_ms
            and candidate.quality >= point.quality
            and (candidate.hybrid_prefill_ms < point.hybrid_prefill_ms or candidate.quality > point.quality)
            for candidate in feasible
        )
        if not dominated:
            frontier.append(point)
    return frontier


def select_fastest_within_budget(points: Iterable[ProfilePoint], tolerance_relative: float) -> ProfilePoint:
    feasible = [point for point in points if point.quality_loss_relative <= tolerance_relative]
    if not feasible:
        raise ValueError("No profile point meets the requested quality tolerance")
    return min(feasible, key=lambda item: (item.hybrid_prefill_ms, -item.quality))
