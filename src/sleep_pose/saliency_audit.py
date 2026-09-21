"""Model-agnostic saliency localization audit for detection explanations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SaliencyPolicy:
    min_energy_inside: float = 0.50
    require_peak_inside: bool = True
    min_concentration_lift: float = 2.0
    max_box_area_ratio: float = 0.80


@dataclass(frozen=True)
class SaliencyAudit:
    decision: str
    reasons: tuple[str, ...]
    width: int
    height: int
    target_box: tuple[int, int, int, int]
    box_area_ratio: float
    energy_inside_ratio: float
    peak_inside: bool
    peak_location: tuple[int, int]
    concentration_lift: float

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["schema_version"] = 1
        return values


def _validate_policy(policy: SaliencyPolicy) -> None:
    for name in ("min_energy_inside", "max_box_area_ratio"):
        value = getattr(policy, name)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    if not math.isfinite(policy.min_concentration_lift) or policy.min_concentration_lift <= 0:
        raise ValueError("min_concentration_lift must be finite and positive")


def audit_saliency_localization(
    saliency: list[list[float]],
    target_box: tuple[int, int, int, int],
    policy: SaliencyPolicy | None = None,
) -> SaliencyAudit:
    """Audit whether non-negative attribution mass localizes inside a target box.

    Coordinates follow half-open image indexing: ``(x_min, y_min, x_max, y_max)``.
    This accepts a plain nested list so evidence can be audited without a tensor runtime.
    """
    policy = policy or SaliencyPolicy()
    _validate_policy(policy)
    if not saliency or not saliency[0]:
        raise ValueError("saliency must be a non-empty 2D grid")
    width = len(saliency[0])
    if any(not row or len(row) != width for row in saliency):
        raise ValueError("saliency rows must form a non-empty rectangular grid")

    height = len(saliency)
    values: list[tuple[float, int, int]] = []
    for y, row in enumerate(saliency):
        for x, raw_value in enumerate(row):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("saliency values must be real numbers")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("saliency values must be finite and non-negative")
            values.append((value, x, y))

    invalid_coordinate = any(
        isinstance(value, bool) or not isinstance(value, int) for value in target_box
    )
    if len(target_box) != 4 or invalid_coordinate:
        raise TypeError("target_box must contain four integer coordinates")
    x_min, y_min, x_max, y_max = target_box
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError("target_box must be non-empty and inside the saliency grid")

    total_energy = sum(value for value, _, _ in values)
    if total_energy <= 0:
        raise ValueError("saliency must contain positive attribution energy")
    inside_energy = sum(
        value for value, x, y in values if x_min <= x < x_max and y_min <= y < y_max
    )
    energy_inside_ratio = inside_energy / total_energy
    peak_value = max(value for value, _, _ in values)
    peak_candidates = [(x, y) for value, x, y in values if value == peak_value]
    peak_location = min(peak_candidates, key=lambda point: (point[1], point[0]))
    peak_inside = x_min <= peak_location[0] < x_max and y_min <= peak_location[1] < y_max
    box_area_ratio = ((x_max - x_min) * (y_max - y_min)) / (width * height)
    concentration_lift = energy_inside_ratio / box_area_ratio

    reasons: list[str] = []
    if box_area_ratio > policy.max_box_area_ratio:
        reasons.append("target_box_too_large")
    if energy_inside_ratio < policy.min_energy_inside:
        reasons.append("energy_inside_below_floor")
    if policy.require_peak_inside and not peak_inside:
        reasons.append("saliency_peak_outside_target")
    if concentration_lift < policy.min_concentration_lift:
        reasons.append("concentration_lift_below_floor")

    return SaliencyAudit(
        decision="pass" if not reasons else "fail",
        reasons=tuple(reasons),
        width=width,
        height=height,
        target_box=target_box,
        box_area_ratio=box_area_ratio,
        energy_inside_ratio=energy_inside_ratio,
        peak_inside=peak_inside,
        peak_location=peak_location,
        concentration_lift=concentration_lift,
    )
