"""Sleep-pose post-processing and evaluation toolkit."""

from .geometry import class_aware_nms, intersection_over_union
from .metrics import DetectionMetrics, evaluate_detections
from .models import Detection
from .saliency_audit import SaliencyAudit, SaliencyPolicy, audit_saliency_localization
from .temporal import PoseSmoother

__all__ = [
    "Detection",
    "DetectionMetrics",
    "PoseSmoother",
    "SaliencyAudit",
    "SaliencyPolicy",
    "audit_saliency_localization",
    "class_aware_nms",
    "evaluate_detections",
    "intersection_over_union",
]
