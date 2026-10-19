# Recovered MediaPipe landmark heuristic

The graduation-project prototype contained a lightweight orientation rule alongside the
YOLOv8 detector. It uses five MediaPipe Pose landmarks:

| ID | Landmark role in the recovered code |
|---:|---|
| 7 | left ear |
| 8 | right ear |
| 12 | right shoulder |
| 23 | left hip |
| 24 | right hip |

## Decision rule

The body score is the squared image-plane hip span divided by the squared distance from the
right shoulder to the right hip:

```text
body_ratio = distance_2d(landmark_24, landmark_23)^2
             / distance_2d(landmark_12, landmark_24)^2
```

At or above the configured threshold, the result is `supine`. Below it, the relative hip
depth selects `facing_left` or `facing_right`. The absolute depth difference between the two ear
landmarks similarly selects `face_up` or a face-side label.

```python
from sleep_pose.landmark_heuristic import classify_mediapipe_landmarks

results = pose.process(rgb_frame)
if results.pose_landmarks is not None:
    orientation = classify_mediapipe_landmarks(results.pose_landmarks)
    print(orientation.to_dict())
```

The package core intentionally does not import MediaPipe or OpenCV. The adapter accepts the
landmark object shape produced by MediaPipe, while the pure decision function remains easy to
test and deploy independently.

## Validation and failure behavior

- landmarks 7, 8, 12, 23 and 24 are mandatory;
- coordinates and thresholds must be finite;
- the shoulder-to-hip reference distance must be non-zero;
- an optional visibility floor rejects low-confidence landmark sets;
- invalid inputs raise `ValueError` instead of emitting an orientation.

## Calibration boundary

The recovered experiments used body thresholds `0.065`, `0.2` and `0.3` in different webcam,
image and video scripts. The default `0.3` preserves the most frequently repeated offline/video
setting; it is not presented as a universally validated threshold. The face threshold was
consistently `0.08` in the recovered scripts.

Before using these rules on a new camera placement, calibrate both thresholds on a held-out,
subject-separated validation set. Report a confusion matrix and per-class precision/recall, and
evaluate low-light, occlusion, bedding, body-shape and camera-angle slices. MediaPipe normalized
`z` is a relative coordinate, not metric depth, so labels also depend on camera and mirroring
conventions.

The original dataset, videos and labels are not included in this repository. Consequently, the
tests validate deterministic rule behavior and input safety—not the thesis accuracy claim.
