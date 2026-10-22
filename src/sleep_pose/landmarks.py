"""Explainable sleep-pose heuristics over MediaPipe-style landmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

LEFT_EAR = 7
RIGHT_EAR = 8
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
REQUIRED_LANDMARKS = (LEFT_EAR, RIGHT_EAR, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


@dataclass(frozen=True)
class PoseLandmark:
    """Normalized landmark coordinates and detector visibility."""

    x: float
    y: float
    z: float
    visibility: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.visibility)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("landmark values must be numeric")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("landmark values must be finite")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("landmark visibility must be between 0 and 1")


@dataclass(frozen=True)
class PoseHeuristicConfig:
    """Calibratable thresholds recovered from the original prototype."""

    supine_ratio_threshold: float = 0.3
    face_up_depth_threshold: float = 0.08
    minimum_visibility: float = 0.5
    minimum_torso_distance_squared: float = 1e-9

    def __post_init__(self) -> None:
        positive = (
            self.supine_ratio_threshold,
            self.face_up_depth_threshold,
            self.minimum_torso_distance_squared,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in positive
        ):
            raise ValueError("pose thresholds must be finite and positive")
        if (
            isinstance(self.minimum_visibility, bool)
            or not isinstance(self.minimum_visibility, (int, float))
            or not math.isfinite(self.minimum_visibility)
            or not 0.0 <= self.minimum_visibility <= 1.0
        ):
            raise ValueError("minimum_visibility must be between 0 and 1")


@dataclass(frozen=True)
class PoseHeuristicDecision:
    """Classification plus the exact evidence used to reach it."""

    status: str
    body_orientation: str
    head_orientation: str
    hip_span_ratio: float | None
    ear_depth_delta: float | None
    minimum_visibility: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def _indeterminate(reason: str, minimum_visibility: float | None = None) -> PoseHeuristicDecision:
    return PoseHeuristicDecision(
        status="indeterminate",
        body_orientation="indeterminate",
        head_orientation="indeterminate",
        hip_span_ratio=None,
        ear_depth_delta=None,
        minimum_visibility=minimum_visibility,
        reasons=(reason,),
    )


def classify_pose_landmarks(
    landmarks: Mapping[int, PoseLandmark],
    *,
    config: PoseHeuristicConfig | None = None,
) -> PoseHeuristicDecision:
    """Classify body and head orientation with auditable geometric evidence.

    Missing, low-visibility or degenerate evidence returns an indeterminate decision
    instead of guessing a pose.
    """

    settings = config or PoseHeuristicConfig()
    missing = [index for index in REQUIRED_LANDMARKS if index not in landmarks]
    if missing:
        return _indeterminate("missing_required_landmarks")

    selected = {index: landmarks[index] for index in REQUIRED_LANDMARKS}
    if any(not isinstance(item, PoseLandmark) for item in selected.values()):
        raise TypeError("landmarks must contain PoseLandmark values")
    minimum_visibility = min(item.visibility for item in selected.values())
    if minimum_visibility < settings.minimum_visibility:
        return _indeterminate("insufficient_landmark_visibility", minimum_visibility)

    left_ear = selected[LEFT_EAR]
    right_ear = selected[RIGHT_EAR]
    shoulder = selected[RIGHT_SHOULDER]
    left_hip = selected[LEFT_HIP]
    right_hip = selected[RIGHT_HIP]

    torso_distance_squared = (shoulder.x - right_hip.x) ** 2 + (shoulder.y - right_hip.y) ** 2
    if torso_distance_squared <= settings.minimum_torso_distance_squared:
        return _indeterminate("degenerate_torso_geometry", minimum_visibility)

    hip_span_squared = (right_hip.x - left_hip.x) ** 2 + (right_hip.y - left_hip.y) ** 2
    hip_span_ratio = hip_span_squared / torso_distance_squared
    if hip_span_ratio >= settings.supine_ratio_threshold:
        body_orientation = "supine"
        body_reason = "hip_span_ratio_at_or_above_threshold"
    elif right_hip.z > left_hip.z:
        body_orientation = "facing_right"
        body_reason = "right_hip_deeper_than_left"
    else:
        body_orientation = "facing_left"
        body_reason = "right_hip_not_deeper_than_left"

    ear_depth_delta = abs(left_ear.z - right_ear.z)
    if ear_depth_delta < settings.face_up_depth_threshold:
        head_orientation = "face_up"
        head_reason = "ear_depth_delta_below_threshold"
    elif right_ear.z < left_ear.z:
        head_orientation = "face_left"
        head_reason = "right_ear_shallower_than_left"
    else:
        head_orientation = "face_right"
        head_reason = "right_ear_not_shallower_than_left"

    return PoseHeuristicDecision(
        status="classified",
        body_orientation=body_orientation,
        head_orientation=head_orientation,
        hip_span_ratio=hip_span_ratio,
        ear_depth_delta=ear_depth_delta,
        minimum_visibility=minimum_visibility,
        reasons=(body_reason, head_reason),
    )


def landmarks_from_mediapipe(items: Sequence[object]) -> dict[int, PoseLandmark]:
    """Extract only the five required landmarks from a MediaPipe landmark sequence."""

    if len(items) <= max(REQUIRED_LANDMARKS):
        raise ValueError("MediaPipe result is missing required landmark indices")
    extracted: dict[int, PoseLandmark] = {}
    for index in REQUIRED_LANDMARKS:
        item = items[index]
        try:
            extracted[index] = PoseLandmark(
                x=item.x,
                y=item.y,
                z=item.z,
                visibility=item.visibility,
            )
        except AttributeError as error:
            raise ValueError("MediaPipe landmark is missing a required field") from error
    return extracted
