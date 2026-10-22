# Sleep Pose Estimation Toolkit

A testable computer-vision support package for sleep-pose detection projects. It covers the
parts that are often hidden behind a notebook: label validation, image-space conversion,
class-aware non-maximum suppression, detection matching and stable frame-to-frame pose output.

## Included

- strict detection and bounding-box validation
- YOLO normalized-label parsing with line-specific error messages
- deterministic class-aware non-maximum suppression
- class-aware greedy matching at a configurable IoU threshold
- per-class precision, recall and F1 plus macro summaries
- confidence filtering and hysteresis-based temporal pose smoothing
- explainable MediaPipe landmark heuristics with geometric evidence and fail-closed gating
- dependency-free CLI demonstration, tests and Python 3.11–3.13 CI
- optional `ultralytics` and headless OpenCV inference dependencies

## Run it

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
pytest
sleep-pose
```

For a YOLOv8 inference adapter:

```bash
pip install -e '.[inference]'
```

## Evaluation boundary

Predictions are sorted by confidence and greedily matched to unused ground-truth boxes of the
same class. A ground-truth box can therefore contribute to at most one true positive. This avoids
the common evaluation bug where duplicate detections inflate recall.

Temporal smoothing maintains a bounded vote window and requires a candidate pose to persist
before changing the reported state. Low-confidence frames do not force a transition.

## Explainable landmark classification

The recovered MediaPipe prototype is available as a dependency-free policy function rather than
camera-bound script code. It classifies body and head orientation from landmarks 7, 8, 12, 23
and 24, and returns the hip-span ratio, ear-depth delta, minimum visibility and stable reason
codes used for the decision. Missing landmarks, low visibility and degenerate torso geometry
produce an `indeterminate` result instead of a guessed pose.

```python
from sleep_pose import classify_pose_landmarks, landmarks_from_mediapipe

landmarks = landmarks_from_mediapipe(results.pose_landmarks.landmark)
decision = classify_pose_landmarks(landmarks)
print(decision.to_dict())
```

The default `0.30` body-ratio and `0.08` head-depth thresholds match the reusable video/image
paths in the historical prototype. They are calibration inputs, not accuracy claims, and should
be selected against a held-out, subject-separated dataset before clinical or safety-sensitive
use.

## Historical project context

The original graduation project used YOLOv8 for sleep-pose detection over 14,715 annotated
images and reported approximately 0.978 mAP. That trained model and dataset are not redistributed
here. This repository provides independently runnable engineering code around the evaluation and
post-processing boundary rather than pretending synthetic tests reproduce the original result.
