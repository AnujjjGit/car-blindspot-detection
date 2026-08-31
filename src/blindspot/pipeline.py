from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from blindspot.geometry import (
    Box,
    NormalizedZone,
    detection_overlap_ratio,
    is_in_blindspot,
)

# COCO ids for road users most relevant to blind-spot monitoring.
DEFAULT_CLASSES = (0, 1, 2, 3, 5, 7)


@dataclass(slots=True)
class PipelineConfig:
    model: str = "yolo11n.pt"
    confidence: float = 0.35
    min_overlap: float = 0.15
    target_classes: tuple[int, ...] = DEFAULT_CLASSES
    zone: NormalizedZone = NormalizedZone()


def _label(class_name: str, confidence: float, track_id: int | None, risky: bool) -> str:
    identity = f" id={track_id}" if track_id is not None else ""
    state = " BLIND-SPOT" if risky else ""
    return f"{class_name} {confidence:.2f}{identity}{state}"


def process_video(input_path: Path, output_path: Path, config: PipelineConfig) -> None:
    model = YOLO(config.model)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    zone = config.zone.to_pixels(width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"unable to create output video: {output_path}")

    class_names = model.names

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=config.confidence,
                classes=list(config.target_classes),
            )

            cv2.rectangle(
                frame,
                (int(zone.x1), int(zone.y1)),
                (int(zone.x2), int(zone.y2)),
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                "Blind-spot zone",
                (int(zone.x1), max(24, int(zone.y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

                for index, raw_box in enumerate(boxes):
                    x1, y1, x2, y2 = raw_box.xyxy[0].cpu().tolist()
                    confidence = float(raw_box.conf[0].cpu())
                    class_id = int(raw_box.cls[0].cpu())
                    track_id = track_ids[index] if index < len(track_ids) else None

                    detection = Box(x1, y1, x2, y2)
                    risky = is_in_blindspot(detection, zone, config.min_overlap)
                    overlap = detection_overlap_ratio(detection, zone)

                    color = (0, 0, 255) if risky else (0, 255, 0)
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    text = _label(class_names[class_id], confidence, track_id, risky)
                    cv2.putText(
                        frame,
                        f"{text} overlap={overlap:.2f}",
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time blind-spot video detection")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--min-overlap", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        model=args.model,
        confidence=args.confidence,
        min_overlap=args.min_overlap,
    )
    process_video(args.input, args.output, config)


if __name__ == "__main__":
    main()
