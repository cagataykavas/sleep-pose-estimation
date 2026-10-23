from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from sleep_pose.calibration import (
    PoseCalibrationRecord,
    calibrate_subject_split,
    load_jsonl,
    main,
)


def record(
    record_id: str,
    subject_id: str,
    *,
    ratio: float,
    hip_depth: float,
    ear_depth: float,
    ear_signed: float,
    body: str,
    head: str,
) -> PoseCalibrationRecord:
    return PoseCalibrationRecord(
        record_id=record_id,
        subject_id=subject_id,
        hip_span_ratio=ratio,
        hip_depth_delta=hip_depth,
        ear_depth_delta=ear_depth,
        ear_depth_signed=ear_signed,
        body_label=body,
        head_label=head,
    )


def dataset() -> list[PoseCalibrationRecord]:
    return [
        record(
            "c1",
            "cal-a",
            ratio=0.40,
            hip_depth=0.0,
            ear_depth=0.02,
            ear_signed=0.02,
            body="supine",
            head="face_up",
        ),
        record(
            "c2",
            "cal-a",
            ratio=0.12,
            hip_depth=0.2,
            ear_depth=0.20,
            ear_signed=-0.20,
            body="facing_right",
            head="face_left",
        ),
        record(
            "c3",
            "cal-b",
            ratio=0.10,
            hip_depth=-0.2,
            ear_depth=0.18,
            ear_signed=0.18,
            body="facing_left",
            head="face_right",
        ),
        record(
            "c4",
            "cal-b",
            ratio=0.35,
            hip_depth=0.0,
            ear_depth=0.03,
            ear_signed=-0.03,
            body="supine",
            head="face_up",
        ),
        record(
            "v1",
            "val-a",
            ratio=0.38,
            hip_depth=0.0,
            ear_depth=0.04,
            ear_signed=0.04,
            body="supine",
            head="face_up",
        ),
        record(
            "v2",
            "val-a",
            ratio=0.08,
            hip_depth=0.2,
            ear_depth=0.16,
            ear_signed=-0.16,
            body="facing_right",
            head="face_left",
        ),
        record(
            "v3",
            "val-b",
            ratio=0.09,
            hip_depth=-0.2,
            ear_depth=0.17,
            ear_signed=0.17,
            body="facing_left",
            head="face_right",
        ),
    ]


def calibrate(records=None):
    return calibrate_subject_split(
        dataset() if records is None else records,
        calibration_subjects={"cal-a", "cal-b"},
        validation_subjects={"val-a", "val-b"},
    )


def test_calibrates_only_on_named_subjects_and_reports_held_out_metrics():
    result = calibrate()

    assert 0.12 < result.supine_ratio_threshold < 0.35
    assert 0.03 < result.face_up_depth_threshold < 0.18
    assert result.calibration_body.accuracy == 1.0
    assert result.calibration_head.accuracy == 1.0
    assert result.validation_body.correct == 3
    assert result.validation_head.correct == 3
    assert result.validation_body.confusion_matrix["supine"]["supine"] == 1


def test_validation_labels_cannot_influence_selected_thresholds():
    original = calibrate()
    changed = [
        replace(item, body_label="supine", head_label="face_up")
        if item.subject_id.startswith("val-")
        else item
        for item in dataset()
    ]
    recalibrated = calibrate(changed)

    assert recalibrated.supine_ratio_threshold == original.supine_ratio_threshold
    assert recalibrated.face_up_depth_threshold == original.face_up_depth_threshold
    assert recalibrated.validation_body.accuracy < original.validation_body.accuracy


def test_order_does_not_change_thresholds_metrics_or_digest():
    forward = calibrate()
    reverse = calibrate(list(reversed(dataset())))

    assert reverse.to_dict() == forward.to_dict()
    assert len(forward.evidence_sha256) == 64


@pytest.mark.parametrize(
    "calibration_subjects,validation_subjects,message",
    [
        ({"cal-a"}, {"cal-a", "val-a", "cal-b", "val-b"}, "leakage"),
        (set(), {"cal-a", "cal-b", "val-a", "val-b"}, "non-empty"),
        ({"cal-a", "cal-b", "missing"}, {"val-a", "val-b"}, "no records"),
    ],
)
def test_rejects_invalid_subject_partitions(calibration_subjects, validation_subjects, message):
    with pytest.raises(ValueError, match=message):
        calibrate_subject_split(
            dataset(),
            calibration_subjects=calibration_subjects,
            validation_subjects=validation_subjects,
        )


def test_rejects_unassigned_subjects_and_duplicate_records():
    extra = dataset() + [replace(dataset()[0], record_id="extra", subject_id="other")]
    with pytest.raises(ValueError, match="unassigned"):
        calibrate(extra)
    with pytest.raises(ValueError, match="unique"):
        calibrate(dataset() + [dataset()[0]])


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"hip_span_ratio": float("nan")}, "finite"),
        ({"hip_span_ratio": -0.1}, "non-negative"),
        ({"ear_depth_delta": 0.2, "ear_depth_signed": 0.1}, "must equal"),
        ({"body_label": "unknown"}, "body_label"),
        ({"head_label": "unknown"}, "head_label"),
    ],
)
def test_rejects_invalid_calibration_evidence(kwargs, message):
    values = {
        "record_id": "x",
        "subject_id": "s",
        "hip_span_ratio": 0.3,
        "hip_depth_delta": 0.1,
        "ear_depth_delta": 0.1,
        "ear_depth_signed": 0.1,
        "body_label": "supine",
        "head_label": "face_right",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PoseCalibrationRecord(**values)


def test_extreme_finite_evidence_does_not_create_infinite_thresholds():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    records = [
        record(
            "c",
            "cal",
            ratio=largest,
            hip_depth=0.0,
            ear_depth=largest,
            ear_signed=largest,
            body="supine",
            head="face_right",
        ),
        record(
            "v",
            "val",
            ratio=0.1,
            hip_depth=-0.1,
            ear_depth=0.1,
            ear_signed=0.1,
            body="facing_left",
            head="face_right",
        ),
    ]

    result = calibrate_subject_split(
        records,
        calibration_subjects={"cal"},
        validation_subjects={"val"},
    )

    assert result.supine_ratio_threshold < largest
    assert result.face_up_depth_threshold < largest


def test_jsonl_loader_is_strict_and_cli_emits_provenance(tmp_path, capsys):
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(json.dumps(asdict(item)) for item in dataset()),
        encoding="utf-8",
    )

    assert len(load_jsonl(source)) == 7
    main(
        [
            str(source),
            "--calibration-subject",
            "cal-a",
            "--calibration-subject",
            "cal-b",
            "--validation-subject",
            "val-a",
            "--validation-subject",
            "val-b",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["validation"]["body"]["accuracy"] == 1.0
    assert len(output["evidence_sha256"]) == 64


def test_jsonl_loader_rejects_blank_unknown_and_empty_records(tmp_path):
    blank = tmp_path / "blank.jsonl"
    blank.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_jsonl(blank)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_jsonl(empty)
