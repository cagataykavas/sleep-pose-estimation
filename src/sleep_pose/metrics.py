"""Greedy class-aware detection matching and per-class quality metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .geometry import intersection_over_union
from .models import Detection


@dataclass(frozen=True)
class ClassMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class DetectionMetrics:
    per_class: dict[str, ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "per_class": {label: asdict(values) for label, values in self.per_class.items()},
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_detections(
    predictions: list[Detection],
    ground_truth: list[Detection],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    labels = sorted({item.label for item in predictions + ground_truth})
    per_class: dict[str, ClassMetrics] = {}

    for label in labels:
        predicted = sorted(
            (item for item in predictions if item.label == label),
            key=lambda item: -item.confidence,
        )
        expected = [item for item in ground_truth if item.label == label]
        unmatched = set(range(len(expected)))
        true_positives = 0
        false_positives = 0
        for prediction in predicted:
            candidates = [
                (intersection_over_union(prediction, expected[index]), index) for index in unmatched
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                true_positives += 1
                unmatched.remove(best_index)
            else:
                false_positives += 1
        false_negatives = len(unmatched)
        precision = _safe_ratio(true_positives, true_positives + false_positives)
        recall = _safe_ratio(true_positives, true_positives + false_negatives)
        f1 = _safe_ratio(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
        per_class[label] = ClassMetrics(
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1,
        )

    count = len(per_class)
    return DetectionMetrics(
        per_class=per_class,
        macro_precision=sum(item.precision for item in per_class.values()) / count
        if count
        else 0.0,
        macro_recall=sum(item.recall for item in per_class.values()) / count if count else 0.0,
        macro_f1=sum(item.f1 for item in per_class.values()) / count if count else 0.0,
    )
