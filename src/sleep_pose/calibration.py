"""Leakage-resistant calibration for the explainable landmark policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BODY_LABELS = ("supine", "facing_left", "facing_right")
HEAD_LABELS = ("face_up", "face_left", "face_right")
_RECORD_FIELDS = {
    "record_id",
    "subject_id",
    "hip_span_ratio",
    "hip_depth_delta",
    "ear_depth_delta",
    "ear_depth_signed",
    "body_label",
    "head_label",
}


@dataclass(frozen=True, slots=True)
class PoseCalibrationRecord:
    """One labelled landmark-evidence record with an explicit subject identity."""

    record_id: str
    subject_id: str
    hip_span_ratio: float
    hip_depth_delta: float
    ear_depth_delta: float
    ear_depth_signed: float
    body_label: str
    head_label: str

    def __post_init__(self) -> None:
        for name in ("record_id", "subject_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
                raise ValueError(f"{name} must be 1-256 UTF-8 bytes")
        numeric = (
            self.hip_span_ratio,
            self.hip_depth_delta,
            self.ear_depth_delta,
            self.ear_depth_signed,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("calibration evidence must contain finite numeric values")
        if self.hip_span_ratio < 0 or self.ear_depth_delta < 0:
            raise ValueError("ratio and absolute depth evidence must be non-negative")
        if not math.isclose(self.ear_depth_delta, abs(self.ear_depth_signed), abs_tol=1e-12):
            raise ValueError("ear_depth_delta must equal abs(ear_depth_signed)")
        if self.body_label not in BODY_LABELS:
            raise ValueError(f"body_label must be one of {BODY_LABELS}")
        if self.head_label not in HEAD_LABELS:
            raise ValueError(f"head_label must be one of {HEAD_LABELS}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PoseCalibrationRecord:
        """Parse a strict record schema so silently ignored fields cannot alter provenance."""

        if not isinstance(value, Mapping) or set(value) != _RECORD_FIELDS:
            raise ValueError(f"calibration record fields must be exactly {sorted(_RECORD_FIELDS)}")
        return cls(**{field: value[field] for field in _RECORD_FIELDS})


@dataclass(frozen=True, slots=True)
class ClassScore:
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    record_count: int
    correct: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassScore]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": {label: asdict(score) for label, score in self.per_class.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True, slots=True)
class SubjectSplitCalibration:
    supine_ratio_threshold: float
    face_up_depth_threshold: float
    calibration_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    calibration_body: ClassificationReport
    calibration_head: ClassificationReport
    validation_body: ClassificationReport
    validation_head: ClassificationReport
    evidence_sha256: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thresholds": {
                "supine_ratio": self.supine_ratio_threshold,
                "face_up_depth": self.face_up_depth_threshold,
            },
            "subjects": {
                "calibration": list(self.calibration_subjects),
                "validation": list(self.validation_subjects),
            },
            "calibration": {
                "body": self.calibration_body.to_dict(),
                "head": self.calibration_head.to_dict(),
            },
            "validation": {
                "body": self.validation_body.to_dict(),
                "head": self.validation_head.to_dict(),
            },
            "evidence_sha256": self.evidence_sha256,
        }


def _predict_body(record: PoseCalibrationRecord, threshold: float) -> str:
    if record.hip_span_ratio >= threshold:
        return "supine"
    return "facing_right" if record.hip_depth_delta > 0 else "facing_left"


def _predict_head(record: PoseCalibrationRecord, threshold: float) -> str:
    if record.ear_depth_delta < threshold:
        return "face_up"
    return "face_left" if record.ear_depth_signed < 0 else "face_right"


def _report(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> ClassificationReport:
    matrix = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for truth, prediction in zip(expected, predicted, strict=True):
        matrix[truth][prediction] += 1

    per_class: dict[str, ClassScore] = {}
    for label in labels:
        true_positives = matrix[label][label]
        false_positives = sum(matrix[truth][label] for truth in labels if truth != label)
        false_negatives = sum(matrix[label][guess] for guess in labels if guess != label)
        support = sum(matrix[label].values())
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = true_positives / support if support else 0.0
        f1 = (
            2 * true_positives / (2 * true_positives + false_positives + false_negatives)
            if 2 * true_positives + false_positives + false_negatives
            else 0.0
        )
        per_class[label] = ClassScore(
            support=support,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    count = len(expected)
    correct = sum(
        truth == prediction for truth, prediction in zip(expected, predicted, strict=True)
    )
    return ClassificationReport(
        record_count=count,
        correct=correct,
        accuracy=correct / count if count else 0.0,
        macro_f1=sum(score.f1 for score in per_class.values()) / len(labels),
        per_class=per_class,
        confusion_matrix=matrix,
    )


def _body_report(
    records: Sequence[PoseCalibrationRecord], threshold: float
) -> ClassificationReport:
    return _report(
        [record.body_label for record in records],
        [_predict_body(record, threshold) for record in records],
        BODY_LABELS,
    )


def _head_report(
    records: Sequence[PoseCalibrationRecord], threshold: float
) -> ClassificationReport:
    return _report(
        [record.head_label for record in records],
        [_predict_head(record, threshold) for record in records],
        HEAD_LABELS,
    )


def _threshold_candidates(values: Iterable[float]) -> tuple[float, ...]:
    unique = sorted(set(values))
    if not unique:
        raise ValueError("threshold calibration requires at least one record")
    candidates = [math.nextafter(unique[0], 0.0) if unique[0] > 0 else math.nextafter(0.0, 1.0)]
    candidates.extend(left / 2 + right / 2 for left, right in zip(unique, unique[1:], strict=False))
    upper = math.nextafter(unique[-1], math.inf)
    if math.isfinite(upper):
        candidates.append(upper)
    return tuple(sorted(set(candidates)))


def _select_threshold(
    records: Sequence[PoseCalibrationRecord],
    *,
    evidence: str,
    default: float,
) -> float:
    if evidence == "body":
        candidates = _threshold_candidates(record.hip_span_ratio for record in records)
        return min(
            candidates,
            key=lambda threshold: (
                -_body_report(records, threshold).macro_f1,
                abs(threshold - default),
                threshold,
            ),
        )
    elif evidence == "head":
        candidates = _threshold_candidates(record.ear_depth_delta for record in records)
        return min(
            candidates,
            key=lambda threshold: (
                -_head_report(records, threshold).macro_f1,
                abs(threshold - default),
                threshold,
            ),
        )
    raise AssertionError("unsupported evidence type")  # pragma: no cover - internal invariant


def calibrate_subject_split(
    records: Sequence[PoseCalibrationRecord],
    *,
    calibration_subjects: Iterable[str],
    validation_subjects: Iterable[str],
) -> SubjectSplitCalibration:
    """Fit thresholds on calibration subjects and report untouched validation performance."""

    calibration_ids = tuple(sorted(set(calibration_subjects)))
    validation_ids = tuple(sorted(set(validation_subjects)))
    if not calibration_ids or not validation_ids:
        raise ValueError("calibration and validation subject sets must both be non-empty")
    overlap = set(calibration_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"subject leakage detected: {sorted(overlap)}")
    if any(
        not isinstance(subject, str) or not subject for subject in calibration_ids + validation_ids
    ):
        raise ValueError("subject IDs must be non-empty strings")
    if not records:
        raise ValueError("calibration requires records")
    if any(not isinstance(record, PoseCalibrationRecord) for record in records):
        raise TypeError("records must contain PoseCalibrationRecord values")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record_id values must be unique")

    assigned = set(calibration_ids) | set(validation_ids)
    observed = {record.subject_id for record in records}
    unknown = observed - assigned
    missing = assigned - observed
    if unknown:
        raise ValueError(f"records contain unassigned subjects: {sorted(unknown)}")
    if missing:
        raise ValueError(f"configured subjects have no records: {sorted(missing)}")

    calibration = [record for record in records if record.subject_id in calibration_ids]
    validation = [record for record in records if record.subject_id in validation_ids]
    supine_threshold = _select_threshold(calibration, evidence="body", default=0.3)
    face_up_threshold = _select_threshold(calibration, evidence="head", default=0.08)

    payload = {
        "schema_version": 1,
        "records": [asdict(record) for record in sorted(records, key=lambda item: item.record_id)],
        "subjects": {"calibration": calibration_ids, "validation": validation_ids},
        "thresholds": {
            "supine_ratio": supine_threshold,
            "face_up_depth": face_up_threshold,
        },
    }
    evidence_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return SubjectSplitCalibration(
        supine_ratio_threshold=supine_threshold,
        face_up_depth_threshold=face_up_threshold,
        calibration_subjects=calibration_ids,
        validation_subjects=validation_ids,
        calibration_body=_body_report(calibration, supine_threshold),
        calibration_head=_head_report(calibration, face_up_threshold),
        validation_body=_body_report(validation, supine_threshold),
        validation_head=_head_report(validation, face_up_threshold),
        evidence_sha256=evidence_sha256,
    )


def load_jsonl(path: Path) -> list[PoseCalibrationRecord]:
    """Read strict, non-empty JSONL calibration records."""

    records: list[PoseCalibrationRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank records are not allowed")
            try:
                value = json.loads(line)
                records.append(PoseCalibrationRecord.from_dict(value))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"line {line_number}: {error}") from error
    if not records:
        raise ValueError("calibration input must not be empty")
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="strict JSONL landmark-evidence records")
    parser.add_argument("--calibration-subject", action="append", required=True)
    parser.add_argument("--validation-subject", action="append", required=True)
    args = parser.parse_args(argv)
    result = calibrate_subject_split(
        load_jsonl(args.input),
        calibration_subjects=args.calibration_subject,
        validation_subjects=args.validation_subject,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
