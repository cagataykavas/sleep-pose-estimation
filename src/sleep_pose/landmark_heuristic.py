"""Validated MediaPipe landmark heuristic recovered from the thesis prototype.

The classifier deliberately has no MediaPipe or OpenCV dependency.  This keeps the
decision rule deterministic and testable while the adapter accepts MediaPipe's
``NormalizedLandmarkList`` shape through structural typing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

REQUIRED_LANDMARK_IDS = frozenset({7, 8, 12, 23, 24})


class BodyOrientation(StrEnum):
    SUPINE = "supine"
    FACING_LEFT = "facing_left"
    FACING_RIGHT = "facing_right"


class FaceOrientation(StrEnum):
    UP = "face_up"
    LEFT = "face_left"
    RIGHT = "face_right"


@dataclass(frozen=True)
class Landmark:
    """One normalized pose landmark used by the heuristic."""

    x: float
    y: float
    z: float
    visibility: float = 1.0

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.visibility)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("landmark coordinates and visibility must be finite")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("landmark visibility must be between 0 and 1")


@dataclass(frozen=True)
class HeuristicThresholds:
    """Calibratable decision thresholds.

    The defaults reproduce the most frequently repeated settings in the recovered
    offline/video scripts. They are historical baselines, not universal constants.
    """

    body_supine_ratio: float = 0.3
    face_up_depth_delta: float = 0.08
    minimum_visibility: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.body_supine_ratio) or self.body_supine_ratio <= 0:
            raise ValueError("body_supine_ratio must be finite and greater than zero")
        if not math.isfinite(self.face_up_depth_delta) or self.face_up_depth_delta < 0:
            raise ValueError("face_up_depth_delta must be finite and non-negative")
        if not math.isfinite(self.minimum_visibility) or not 0 <= self.minimum_visibility <= 1:
            raise ValueError("minimum_visibility must be between 0 and 1")


@dataclass(frozen=True)
class PoseHeuristicResult:
    body: BodyOrientation
    face: FaceOrientation
    body_ratio: float
    face_depth_delta: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            **asdict(self),
            "body": self.body.value,
            "face": self.face.value,
        }


class _LandmarkLike(Protocol):
    x: float
    y: float
    z: float


class _PoseLandmarksLike(Protocol):
    landmark: Sequence[_LandmarkLike]


def classify_landmarks(
    landmarks: Mapping[int, Landmark],
    *,
    thresholds: HeuristicThresholds | None = None,
) -> PoseHeuristicResult:
    """Classify body and face orientation from five MediaPipe pose landmarks.

    Body orientation uses the squared 2D hip-span to right-shoulder/right-hip
    distance ratio from the thesis prototype. Side direction uses hip depth;
    face direction uses the depth difference between landmarks 7 and 8.
    """

    config = thresholds or HeuristicThresholds()
    missing = sorted(REQUIRED_LANDMARK_IDS.difference(landmarks))
    if missing:
        raise ValueError(f"missing required landmark ids: {missing}")

    selected = {identifier: landmarks[identifier] for identifier in REQUIRED_LANDMARK_IDS}
    low_visibility = sorted(
        identifier
        for identifier, landmark in selected.items()
        if landmark.visibility < config.minimum_visibility
    )
    if low_visibility:
        raise ValueError(f"landmarks below minimum_visibility: {low_visibility}")

    ear_left = selected[7]
    ear_right = selected[8]
    shoulder_right = selected[12]
    hip_left = selected[23]
    hip_right = selected[24]

    hip_span = math.hypot(hip_right.x - hip_left.x, hip_right.y - hip_left.y)
    body_reference = math.hypot(
        shoulder_right.x - hip_right.x,
        shoulder_right.y - hip_right.y,
    )
    if body_reference == 0.0:
        raise ValueError("landmarks 12 and 24 must not share the same x/y coordinates")

    body_ratio = (hip_span / body_reference) ** 2
    if not math.isfinite(body_ratio):
        raise ValueError("body ratio must be finite")

    if body_ratio >= config.body_supine_ratio:
        body = BodyOrientation.SUPINE
    elif hip_right.z > hip_left.z:
        body = BodyOrientation.FACING_RIGHT
    else:
        body = BodyOrientation.FACING_LEFT

    face_depth_delta = abs(ear_left.z - ear_right.z)
    if face_depth_delta < config.face_up_depth_delta:
        face = FaceOrientation.UP
    elif ear_right.z < ear_left.z:
        face = FaceOrientation.LEFT
    else:
        face = FaceOrientation.RIGHT

    return PoseHeuristicResult(
        body=body,
        face=face,
        body_ratio=body_ratio,
        face_depth_delta=face_depth_delta,
    )


def classify_mediapipe_landmarks(
    pose_landmarks: _PoseLandmarksLike,
    *,
    thresholds: HeuristicThresholds | None = None,
) -> PoseHeuristicResult:
    """Adapt a MediaPipe pose-landmark list to :func:`classify_landmarks`."""

    try:
        raw_landmarks = pose_landmarks.landmark
    except AttributeError as error:
        raise ValueError("pose_landmarks must expose a landmark sequence") from error
    if len(raw_landmarks) <= max(REQUIRED_LANDMARK_IDS):
        raise ValueError("pose_landmarks must contain at least 25 landmarks")

    converted: dict[int, Landmark] = {}
    for identifier in REQUIRED_LANDMARK_IDS:
        raw: Any = raw_landmarks[identifier]
        try:
            converted[identifier] = Landmark(
                x=float(raw.x),
                y=float(raw.y),
                z=float(raw.z),
                visibility=float(getattr(raw, "visibility", 1.0)),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid landmark at id {identifier}") from error

    return classify_landmarks(converted, thresholds=thresholds)
