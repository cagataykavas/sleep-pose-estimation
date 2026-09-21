import json

import pytest

from sleep_pose.saliency_audit import SaliencyPolicy, audit_saliency_localization


def test_passes_localized_explanation_and_is_json_ready():
    saliency = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 3.0, 0.0],
        [0.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    report = audit_saliency_localization(saliency, (1, 1, 3, 3))

    assert report.decision == "pass"
    assert report.energy_inside_ratio == 1.0
    assert report.peak_inside is True
    assert report.concentration_lift == 4.0
    assert json.loads(json.dumps(report.to_dict()))["schema_version"] == 1


def test_fails_diffuse_map_using_area_normalized_lift():
    report = audit_saliency_localization(
        [[1.0, 1.0, 1.0, 1.0] for _ in range(4)],
        (1, 1, 3, 3),
        SaliencyPolicy(min_energy_inside=0.2, require_peak_inside=False),
    )

    assert report.energy_inside_ratio == 0.25
    assert report.concentration_lift == 1.0
    assert report.reasons == ("concentration_lift_below_floor",)


def test_fails_when_peak_is_outside_despite_enough_box_energy():
    saliency = [
        [6.0, 0.0, 0.0],
        [0.0, 4.0, 4.0],
        [0.0, 4.0, 4.0],
    ]
    report = audit_saliency_localization(
        saliency,
        (1, 1, 3, 3),
        SaliencyPolicy(min_energy_inside=0.5, min_concentration_lift=1.0),
    )

    assert report.energy_inside_ratio > 0.5
    assert report.peak_inside is False
    assert "saliency_peak_outside_target" in report.reasons


def test_rejects_large_box_that_can_trivially_capture_energy():
    report = audit_saliency_localization(
        [[1.0] * 4 for _ in range(4)],
        (0, 0, 4, 4),
        SaliencyPolicy(min_energy_inside=0.5, min_concentration_lift=1.0),
    )

    assert report.decision == "fail"
    assert "target_box_too_large" in report.reasons


@pytest.mark.parametrize(
    ("saliency", "box", "error"),
    [
        ([], (0, 0, 1, 1), "non-empty"),
        ([[1.0], [1.0, 2.0]], (0, 0, 1, 1), "rectangular"),
        ([[0.0]], (0, 0, 1, 1), "positive attribution"),
        ([[float("nan")]], (0, 0, 1, 1), "finite and non-negative"),
        ([[-1.0]], (0, 0, 1, 1), "finite and non-negative"),
        ([[1.0]], (0, 0, 2, 1), "inside"),
    ],
)
def test_rejects_malformed_evidence(saliency, box, error):
    with pytest.raises((TypeError, ValueError), match=error):
        audit_saliency_localization(saliency, box)


def test_tied_peak_location_is_deterministic():
    report = audit_saliency_localization(
        [[2.0, 2.0], [0.0, 0.0]],
        (0, 0, 1, 1),
        SaliencyPolicy(min_energy_inside=0.4, min_concentration_lift=1.0),
    )

    assert report.peak_location == (0, 0)
