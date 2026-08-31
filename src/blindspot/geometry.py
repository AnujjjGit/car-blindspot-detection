from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned pixel-space bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class NormalizedZone:
    """Blind-spot rectangle in normalized coordinates (0..1)."""

    x1: float = 0.58
    y1: float = 0.52
    x2: float = 0.98
    y2: float = 0.98

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized zone coordinates must be between 0 and 1")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("zone must have positive width and height")

    def to_pixels(self, width: int, height: int) -> Box:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        return Box(
            x1=self.x1 * width,
            y1=self.y1 * height,
            x2=self.x2 * width,
            y2=self.y2 * height,
        )


def intersection_area(a: Box, b: Box) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def detection_overlap_ratio(detection: Box, zone: Box) -> float:
    """Fraction of the detection box that lies inside the blind-spot zone."""

    if detection.area == 0:
        return 0.0
    return intersection_area(detection, zone) / detection.area


def is_in_blindspot(detection: Box, zone: Box, min_overlap: float = 0.15) -> bool:
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    return detection_overlap_ratio(detection, zone) >= min_overlap
