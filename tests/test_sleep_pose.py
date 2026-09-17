from __future__ import annotations

import pytest

from sleep_pose.geometry import class_aware_nms, intersection_over_union
from sleep_pose.labels import parse_yolo_lines
from sleep_pose.metrics import evaluate_detections
from sleep_pose.models import Detection
from sleep_pose.temporal import PoseSmoother


def box(label="supine", confidence=1.0, offset=0.0):
    return Detection(label, offset, offset, 100 + offset, 100 + offset, confidence)


def test_detection_validation():
    with pytest.raises(ValueError, match="positive width"):
        Detection("supine", 10, 10, 5, 20)
    with pytest.raises(ValueError, match="confidence"):
        box(confidence=1.1)


def test_iou_identity_and_disjoint():
    assert intersection_over_union(box(), box()) == pytest.approx(1.0)
    assert intersection_over_union(box(), box(offset=200)) == 0.0


def test_iou_partial_overlap():
    assert intersection_over_union(box(), box(offset=50)) == pytest.approx(2500 / 17500)


def test_nms_suppresses_same_class_overlap():
    result = class_aware_nms([box(confidence=0.9), box(confidence=0.8, offset=2)])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_nms_keeps_different_classes():
    result = class_aware_nms([box("supine", 0.9), box("side", 0.8)])
    assert len(result) == 2


def test_nms_filters_low_confidence():
    assert class_aware_nms([box(confidence=0.1)], confidence_threshold=0.25) == []


def test_yolo_parser_converts_to_image_space():
    result = parse_yolo_lines(
        ["0 0.5 0.5 0.5 0.25"],
        image_width=200,
        image_height=100,
        classes={0: "supine"},
    )
    assert result == [Detection("supine", 50, 37.5, 150, 62.5)]


@pytest.mark.parametrize(
    "line,message",
    [
        ("0 0.5", "expected 5"),
        ("4 0.5 0.5 0.2 0.2", "unknown class"),
        ("0 1.2 0.5 0.2 0.2", "normalized"),
        ("0 0.5 0.5 0 0.2", "positive"),
    ],
)
def test_yolo_parser_reports_invalid_labels(line, message):
    with pytest.raises(ValueError, match=message):
        parse_yolo_lines([line], image_width=100, image_height=100, classes={0: "supine"})


def test_metrics_match_by_class_and_iou():
    result = evaluate_detections(
        [box("supine", 0.9), box("prone", 0.7, offset=200)],
        [box("supine"), box("side", offset=200)],
    )
    assert result.per_class["supine"].true_positives == 1
    assert result.per_class["prone"].false_positives == 1
    assert result.per_class["side"].false_negatives == 1
    assert result.macro_f1 == pytest.approx(1 / 3)


def test_metrics_do_not_match_one_truth_twice():
    result = evaluate_detections(
        [box(confidence=0.9), box(confidence=0.8, offset=2)],
        [box()],
    )
    assert result.per_class["supine"].true_positives == 1
    assert result.per_class["supine"].false_positives == 1


def test_empty_metrics_are_zero():
    result = evaluate_detections([], [])
    assert result.macro_f1 == 0.0
    assert result.per_class == {}


def test_smoother_requires_votes_and_hysteresis():
    smoother = PoseSmoother(window_size=4, minimum_votes=2, transition_frames=2)
    assert smoother.update([box("supine", 0.9)]) is None
    assert smoother.update([box("supine", 0.9)]) is None
    assert smoother.update([box("supine", 0.9)]) == "supine"


def test_smoother_ignores_low_confidence_noise():
    smoother = PoseSmoother(window_size=3, minimum_votes=2, transition_frames=1)
    assert smoother.update([box("side", 0.2)]) is None
    assert smoother.update([box("side", 0.9)]) is None
    assert smoother.update([box("side", 0.9)]) == "side"


def test_smoother_does_not_flip_on_single_frame():
    smoother = PoseSmoother(window_size=3, minimum_votes=2, transition_frames=1)
    smoother.update([box("supine", 0.9)])
    assert smoother.update([box("supine", 0.9)]) == "supine"
    assert smoother.update([box("side", 0.99)]) == "supine"
