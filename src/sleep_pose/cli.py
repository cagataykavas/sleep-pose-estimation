"""Deterministic command-line demonstration of the evaluation pipeline."""

from __future__ import annotations

import json

from .geometry import class_aware_nms
from .metrics import evaluate_detections
from .models import Detection
from .temporal import PoseSmoother


def main() -> None:
    ground_truth = [
        Detection("supine", 40, 30, 180, 150),
        Detection("side", 220, 40, 360, 160),
    ]
    predictions = [
        Detection("supine", 42, 32, 178, 149, 0.94),
        Detection("supine", 48, 35, 181, 152, 0.71),
        Detection("side", 225, 43, 358, 158, 0.90),
        Detection("prone", 400, 50, 510, 170, 0.55),
    ]
    filtered = class_aware_nms(predictions)
    metrics = evaluate_detections(filtered, ground_truth)
    smoother = PoseSmoother(window_size=5, minimum_votes=3, transition_frames=2)
    sequence = [
        [Detection("supine", 0, 0, 10, 10, 0.9)],
        [Detection("supine", 0, 0, 10, 10, 0.8)],
        [Detection("supine", 0, 0, 10, 10, 0.95)],
        [Detection("supine", 0, 0, 10, 10, 0.92)],
    ]
    states = [smoother.update(frame) for frame in sequence]
    print(
        json.dumps(
            {
                "input_detections": len(predictions),
                "detections_after_nms": len(filtered),
                "metrics": metrics.to_dict(),
                "smoothed_states": states,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
