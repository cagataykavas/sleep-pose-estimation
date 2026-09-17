"""Bounding-box geometry and deterministic class-aware suppression."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import Detection


def intersection_over_union(first: Detection, second: Detection) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(
    detections: Iterable[Detection],
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
) -> list[Detection]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        if detection.confidence >= confidence_threshold:
            grouped[detection.label].append(detection)

    kept: list[Detection] = []
    for label in sorted(grouped):
        candidates = sorted(
            grouped[label],
            key=lambda item: (-item.confidence, item.x1, item.y1, item.x2, item.y2),
        )
        while candidates:
            selected = candidates.pop(0)
            kept.append(selected)
            candidates = [
                candidate
                for candidate in candidates
                if intersection_over_union(selected, candidate) <= iou_threshold
            ]
    return sorted(kept, key=lambda item: (-item.confidence, item.label, item.x1, item.y1))
