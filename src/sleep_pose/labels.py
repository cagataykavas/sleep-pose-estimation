"""YOLO label parsing with image-space conversion and useful validation errors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .models import Detection


def parse_yolo_lines(
    lines: Iterable[str],
    *,
    image_width: int,
    image_height: int,
    classes: Mapping[int, str],
) -> list[Detection]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    detections: list[Detection] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_number}: expected 5 values")
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(f"line {line_number}: invalid numeric value") from error
        if class_id not in classes:
            raise ValueError(f"line {line_number}: unknown class {class_id}")
        if not all(0 <= value <= 1 for value in (center_x, center_y, width, height)):
            raise ValueError(f"line {line_number}: coordinates must be normalized")
        if width == 0 or height == 0:
            raise ValueError(f"line {line_number}: width and height must be positive")

        x1 = (center_x - width / 2) * image_width
        y1 = (center_y - height / 2) * image_height
        x2 = (center_x + width / 2) * image_width
        y2 = (center_y + height / 2) * image_height
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            raise ValueError(f"line {line_number}: box extends outside the image")
        detections.append(Detection(classes[class_id], x1, y1, x2, y2))
    return detections


def load_yolo_labels(
    path: str | Path,
    *,
    image_width: int,
    image_height: int,
    classes: Mapping[int, str],
) -> list[Detection]:
    with Path(path).open(encoding="utf-8") as stream:
        return parse_yolo_lines(
            stream,
            image_width=image_width,
            image_height=image_height,
            classes=classes,
        )
