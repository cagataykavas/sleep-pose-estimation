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

## Historical project context

The original graduation project used YOLOv8 for sleep-pose detection over 14,715 annotated
images and reported approximately 0.978 mAP. That trained model and dataset are not redistributed
here. This repository provides independently runnable engineering code around the evaluation and
post-processing boundary rather than pretending synthetic tests reproduce the original result.


## Saliency localization audit

`audit_saliency_localization` evaluates whether a non-negative attribution map is concentrated inside the annotated pose bounding box. It combines three complementary signals:

- energy-in-box: the fraction of total attribution mass inside the target;
- pointing game: whether the deterministic peak lies inside the target;
- concentration lift: energy-in-box divided by the target's image-area share.

The area-normalized lift prevents a large box from appearing faithful simply because it covers most pixels. A configurable maximum box-area guard also marks trivially broad localization evidence as unsuitable for gating. The report is deterministic and JSON-ready, with explicit reason codes for CI or experiment artifacts.

This audit measures spatial alignment, not causal faithfulness. A heatmap can overlap the annotated person while still relying on spurious features, and bounding boxes include background. Production evaluation should pair localization with deletion/insertion tests, negative controls, multiple explanation methods and subgroup analysis across pose, lighting, occlusion and camera conditions.
