from __future__ import annotations

from types import SimpleNamespace

import pytest

from sleep_pose.landmark_heuristic import (
    BodyOrientation,
    FaceOrientation,
    HeuristicThresholds,
    Landmark,
    classify_landmarks,
    classify_mediapipe_landmarks,
)


def landmarks(
    *,
    hip_span: float = 0.2,
    left_hip_z: float = 0.0,
    right_hip_z: float = 0.0,
    left_ear_z: float = 0.0,
    right_ear_z: float = 0.0,
    visibility: float = 1.0,
) -> dict[int, Landmark]:
    return {
        7: Landmark(0.4, 0.2, left_ear_z, visibility),
        8: Landmark(0.6, 0.2, right_ear_z, visibility),
        12: Landmark(0.5, 0.0, 0.0, visibility),
        23: Landmark(0.5 - hip_span, 1.0, left_hip_z, visibility),
        24: Landmark(0.5, 1.0, right_hip_z, visibility),
    }


def test_classifies_supine_and_face_up() -> None:
    result = classify_landmarks(landmarks(hip_span=0.6))

    assert result.body is BodyOrientation.SUPINE
    assert result.face is FaceOrientation.UP
    assert result.body_ratio == pytest.approx(0.36)
    assert result.face_depth_delta == 0.0


@pytest.mark.parametrize(
    ("left_z", "right_z", "expected"),
    [
        (0.0, 0.2, BodyOrientation.FACING_RIGHT),
        (0.2, 0.0, BodyOrientation.FACING_LEFT),
    ],
)
def test_classifies_side_from_hip_depth(left_z: float, right_z: float, expected) -> None:
    result = classify_landmarks(landmarks(hip_span=0.2, left_hip_z=left_z, right_hip_z=right_z))

    assert result.body is expected


@pytest.mark.parametrize(
    ("left_z", "right_z", "expected"),
    [
        (0.2, 0.0, FaceOrientation.LEFT),
        (0.0, 0.2, FaceOrientation.RIGHT),
    ],
)
def test_classifies_face_side_from_ear_depth(left_z: float, right_z: float, expected) -> None:
    result = classify_landmarks(landmarks(hip_span=0.2, left_ear_z=left_z, right_ear_z=right_z))

    assert result.face is expected


def test_threshold_boundaries_match_recovered_offline_rule() -> None:
    result = classify_landmarks(
        landmarks(hip_span=0.5, left_ear_z=0.08),
        thresholds=HeuristicThresholds(body_supine_ratio=0.25),
    )

    assert result.body is BodyOrientation.SUPINE
    assert result.face is FaceOrientation.LEFT


def test_thresholds_are_explicitly_calibratable() -> None:
    points = landmarks(hip_span=0.2)

    assert classify_landmarks(points).body is BodyOrientation.FACING_LEFT
    assert (
        classify_landmarks(
            points,
            thresholds=HeuristicThresholds(body_supine_ratio=0.03),
        ).body
        is BodyOrientation.SUPINE
    )


def test_missing_required_landmarks_fail_closed() -> None:
    points = landmarks()
    del points[24]

    with pytest.raises(ValueError, match=r"missing required landmark ids: \[24\]"):
        classify_landmarks(points)


def test_zero_length_body_reference_fails_closed() -> None:
    points = landmarks()
    points[12] = Landmark(points[24].x, points[24].y, 0.0)

    with pytest.raises(ValueError, match="must not share"):
        classify_landmarks(points)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_landmarks_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Landmark(value, 0.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"body_supine_ratio": 0.0},
        {"body_supine_ratio": float("inf")},
        {"face_up_depth_delta": -0.1},
        {"minimum_visibility": 1.1},
    ],
)
def test_invalid_threshold_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        HeuristicThresholds(**kwargs)


def test_low_visibility_landmarks_fail_closed_when_guard_is_enabled() -> None:
    with pytest.raises(ValueError, match="below minimum_visibility"):
        classify_landmarks(
            landmarks(visibility=0.4),
            thresholds=HeuristicThresholds(minimum_visibility=0.5),
        )


def test_result_is_json_ready() -> None:
    result = classify_landmarks(landmarks(hip_span=0.6))

    assert result.to_dict() == {
        "body": "supine",
        "face": "face_up",
        "body_ratio": pytest.approx(0.36),
        "face_depth_delta": 0.0,
    }


def test_mediapipe_adapter_accepts_structural_landmark_objects() -> None:
    raw = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(25)]
    for identifier, point in landmarks(hip_span=0.6).items():
        raw[identifier] = SimpleNamespace(**point.__dict__)

    result = classify_mediapipe_landmarks(SimpleNamespace(landmark=raw))

    assert result.body is BodyOrientation.SUPINE
    assert result.face is FaceOrientation.UP


def test_mediapipe_adapter_rejects_short_landmark_lists() -> None:
    with pytest.raises(ValueError, match="at least 25"):
        classify_mediapipe_landmarks(SimpleNamespace(landmark=[]))
