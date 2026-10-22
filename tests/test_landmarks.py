from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from sleep_pose.landmarks import (
    LEFT_EAR,
    LEFT_HIP,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseHeuristicConfig,
    PoseLandmark,
    classify_pose_landmarks,
    landmarks_from_mediapipe,
)


def point(x=0.0, y=0.0, z=0.0, visibility=0.99):
    return PoseLandmark(x, y, z, visibility)


def pose():
    return {
        LEFT_EAR: point(z=0.00),
        RIGHT_EAR: point(z=0.05),
        RIGHT_SHOULDER: point(x=0.7, y=0.2),
        LEFT_HIP: point(x=0.3, y=0.7, z=0.1),
        RIGHT_HIP: point(x=0.7, y=0.7, z=0.1),
    }


def test_classifies_supine_and_face_up_with_auditable_evidence():
    decision = classify_pose_landmarks(pose())

    assert decision.status == "classified"
    assert decision.body_orientation == "supine"
    assert decision.head_orientation == "face_up"
    assert decision.hip_span_ratio == pytest.approx(0.64)
    assert decision.ear_depth_delta == pytest.approx(0.05)
    assert decision.reasons == (
        "hip_span_ratio_at_or_above_threshold",
        "ear_depth_delta_below_threshold",
    )


def test_classifies_right_body_and_left_head_orientation():
    landmarks = pose()
    landmarks[LEFT_HIP] = point(x=0.45, y=0.7, z=0.1)
    landmarks[RIGHT_HIP] = point(x=0.55, y=0.7, z=0.2)
    landmarks[RIGHT_EAR] = point(z=-0.2)

    decision = classify_pose_landmarks(landmarks)

    assert decision.body_orientation == "facing_right"
    assert decision.head_orientation == "face_left"


def test_classifies_left_body_and_right_head_orientation():
    landmarks = pose()
    landmarks[LEFT_HIP] = point(x=0.45, y=0.7, z=0.2)
    landmarks[RIGHT_HIP] = point(x=0.55, y=0.7, z=0.1)
    landmarks[RIGHT_EAR] = point(z=0.2)

    decision = classify_pose_landmarks(landmarks)

    assert decision.body_orientation == "facing_left"
    assert decision.head_orientation == "face_right"


def test_threshold_boundaries_are_explicit_and_deterministic():
    landmarks = pose()
    landmarks[LEFT_HIP] = point(x=0.0, y=0.7)
    landmarks[RIGHT_HIP] = point(x=0.3**0.5 * 0.5, y=0.7)
    landmarks[RIGHT_SHOULDER] = point(x=landmarks[RIGHT_HIP].x, y=0.2)
    landmarks[RIGHT_EAR] = point(z=0.08)

    decision = classify_pose_landmarks(landmarks)

    assert decision.hip_span_ratio == pytest.approx(0.3)
    assert decision.body_orientation == "supine"
    assert decision.head_orientation == "face_right"


def test_missing_landmark_fails_closed():
    landmarks = pose()
    del landmarks[RIGHT_HIP]

    decision = classify_pose_landmarks(landmarks)

    assert decision.status == "indeterminate"
    assert decision.reasons == ("missing_required_landmarks",)


def test_low_visibility_fails_closed_without_geometric_claims():
    landmarks = pose()
    landmarks[LEFT_EAR] = replace(landmarks[LEFT_EAR], visibility=0.49)

    decision = classify_pose_landmarks(landmarks)

    assert decision.status == "indeterminate"
    assert decision.minimum_visibility == pytest.approx(0.49)
    assert decision.hip_span_ratio is None
    assert decision.reasons == ("insufficient_landmark_visibility",)


def test_degenerate_torso_geometry_fails_closed():
    landmarks = pose()
    landmarks[RIGHT_SHOULDER] = point(x=0.7, y=0.7)

    decision = classify_pose_landmarks(landmarks)

    assert decision.status == "indeterminate"
    assert decision.reasons == ("degenerate_torso_geometry",)


def test_rejects_unvalidated_landmark_objects():
    landmarks = pose()
    landmarks[LEFT_EAR] = object()

    with pytest.raises(TypeError, match="PoseLandmark"):
        classify_pose_landmarks(landmarks)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"supine_ratio_threshold": 0}, "finite and positive"),
        ({"face_up_depth_threshold": float("nan")}, "finite and positive"),
        ({"minimum_torso_distance_squared": -1}, "finite and positive"),
        ({"minimum_visibility": -0.1}, "between 0 and 1"),
        ({"minimum_visibility": 1.1}, "between 0 and 1"),
    ],
)
def test_rejects_unsafe_threshold_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PoseHeuristicConfig(**kwargs)


@pytest.mark.parametrize(
    "values,message",
    [
        ((False, 0.0, 0.0, 1.0), "numeric"),
        ((0.0, 0.0, float("inf"), 1.0), "finite"),
        ((0.0, 0.0, 0.0, 1.1), "between 0 and 1"),
    ],
)
def test_rejects_invalid_landmark_values(values, message):
    with pytest.raises(ValueError, match=message):
        PoseLandmark(*values)


def test_extracts_only_required_mediapipe_landmarks():
    items = [SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9) for _ in range(33)]

    extracted = landmarks_from_mediapipe(items)

    assert set(extracted) == {7, 8, 12, 23, 24}
    assert extracted[24] == point(0.1, 0.2, 0.3, 0.9)


def test_rejects_incomplete_or_malformed_mediapipe_results():
    with pytest.raises(ValueError, match="indices"):
        landmarks_from_mediapipe([object()] * 24)
    malformed = [SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9) for _ in range(33)]
    malformed[7] = SimpleNamespace(x=0.1, y=0.2, z=0.3)
    with pytest.raises(ValueError, match="required field"):
        landmarks_from_mediapipe(malformed)


def test_decision_serialization_is_json_ready_and_stable():
    serialized = classify_pose_landmarks(pose()).to_dict()

    assert serialized["status"] == "classified"
    assert serialized["reasons"] == [
        "hip_span_ratio_at_or_above_threshold",
        "ear_depth_delta_below_threshold",
    ]
