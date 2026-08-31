import pytest

from blindspot.geometry import Box, NormalizedZone, detection_overlap_ratio, is_in_blindspot


def test_zone_scales_to_frame_size() -> None:
    zone = NormalizedZone(0.5, 0.5, 1.0, 1.0).to_pixels(1920, 1080)
    assert zone == Box(960.0, 540.0, 1920.0, 1080.0)


def test_full_detection_inside_zone_has_unit_overlap() -> None:
    detection = Box(75, 75, 100, 100)
    zone = Box(50, 50, 100, 100)
    assert detection_overlap_ratio(detection, zone) == pytest.approx(1.0)
    assert is_in_blindspot(detection, zone)


def test_partial_overlap_respects_threshold() -> None:
    detection = Box(0, 0, 100, 100)
    zone = Box(80, 0, 180, 100)
    assert detection_overlap_ratio(detection, zone) == pytest.approx(0.2)
    assert is_in_blindspot(detection, zone, min_overlap=0.15)
    assert not is_in_blindspot(detection, zone, min_overlap=0.25)


def test_non_overlapping_detection_is_safe() -> None:
    assert not is_in_blindspot(Box(0, 0, 10, 10), Box(20, 20, 30, 30))


def test_zero_area_detection_is_safe() -> None:
    assert detection_overlap_ratio(Box(1, 1, 1, 10), Box(0, 0, 20, 20)) == 0.0


def test_invalid_normalized_zone_rejected() -> None:
    with pytest.raises(ValueError):
        NormalizedZone(0.9, 0.2, 0.1, 0.8)
