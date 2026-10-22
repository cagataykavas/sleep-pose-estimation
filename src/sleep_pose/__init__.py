"""Sleep-pose post-processing and evaluation toolkit."""

from .geometry import class_aware_nms, intersection_over_union
from .landmarks import (
    PoseHeuristicConfig,
    PoseHeuristicDecision,
    PoseLandmark,
    classify_pose_landmarks,
    landmarks_from_mediapipe,
)
from .metrics import DetectionMetrics, evaluate_detections
from .models import Detection
from .temporal import PoseSmoother

__all__ = [
    "Detection",
    "DetectionMetrics",
    "PoseHeuristicConfig",
    "PoseHeuristicDecision",
    "PoseLandmark",
    "PoseSmoother",
    "class_aware_nms",
    "classify_pose_landmarks",
    "evaluate_detections",
    "intersection_over_union",
    "landmarks_from_mediapipe",
]
