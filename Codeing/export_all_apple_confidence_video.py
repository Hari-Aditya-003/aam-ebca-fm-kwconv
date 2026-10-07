#!/usr/bin/env python3
"""Render every enhanced-detector apple box with confidence only.

The source detections were produced by the single combined
AAM -> EBCA -> FM -> KWConv detector. No depth threshold is applied, and the
video contains no depth, track ID, count, or counting-line overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from itertools import groupby
from pathlib import Path
from typing import Iterator, TextIO

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO = PROJECT_ROOT / "Dataset" / "DJI_0418.MP4"
DETECTIONS_CSV = PROJECT_ROOT / "Results" / "csv" / "apple_detections_confidence_depth.csv"
OUTPUT_VIDEO = PROJECT_ROOT / "Results" / "inference" / "apple_confidence_all_detections.mp4"
OUTPUT_SUMMARY = PROJECT_ROOT / "Results" / "inference" / "apple_confidence_all_detections_summary.json"


def grouped_rows(handle: TextIO) -> Iterator[tuple[int, list[dict[str, str]]]]:
    rows = csv.DictReader(handle)
    for frame, group in groupby(rows, key=lambda row: int(row["frame"])):
        yield frame, list(group)


class FrameCursor:
    def __init__(self, rows: Iterator[tuple[int, list[dict[str, str]]]]) -> None:
        self.rows = iter(rows)
        self.current = next(self.rows, None)

    def at(self, frame: int) -> list[dict[str, str]]:
        while self.current is not None and self.current[0] < frame:
            self.current = next(self.rows, None)
        if self.current is None or self.current[0] != frame:
            return []
        result = self.current[1]
        self.current = next(self.rows, None)
        return result


def draw_box(image, row: dict[str, str]) -> None:
    colour = (0, 255, 255)
    x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
    x1, x2 = sorted((max(0, x1), min(image.shape[1] - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(image.shape[0] - 1, y2)))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 7, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 4, cv2.LINE_AA)

    label = f"apple C:{float(row['confidence']):.2f}"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    text_x = min(x1 + 1, max(0, image.shape[1] - text_width - 6))
    text_y = max(text_height + baseline + 5, y1 - 7)
    cv2.rectangle(
        image,
        (max(0, text_x - 3), max(0, text_y - text_height - baseline - 4)),
        (min(image.shape[1] - 1, text_x + text_width + 4), min(image.shape[0] - 1, text_y + baseline + 2)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, label, (text_x, text_y), font, scale, colour, thickness, cv2.LINE_AA)


def start_encoder(path: Path, width: int, height: int, fps: float, encoder: str) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.12g}", "-i", "-", "-an", "-c:v", encoder,
    ]
    if encoder == "h264_nvenc":
        command += ["-preset", "p4", "-cq", "19"]
    else:
        command += ["-preset", "medium", "-crf", "19"]
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def export(source: Path, detections_csv: Path, output: Path, encoder: str) -> dict[str, object]:
    for required in (source, detections_csv):
        if not required.is_file():
            raise FileNotFoundError(required)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Could not open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    process = start_encoder(output, width, height, fps, encoder)
    frame_index = 0
    detection_count = 0
    frames_with_detections = 0
    started = time.monotonic()
    try:
        with detections_csv.open(encoding="utf-8", newline="") as handle:
            detections = FrameCursor(grouped_rows(handle))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rows = detections.at(frame_index)
                if rows:
                    frames_with_detections += 1
                detection_count += len(rows)
                for row in rows:
                    draw_box(frame, row)
                if process.stdin is None:
                    raise RuntimeError("Encoder stdin closed unexpectedly")
                process.stdin.write(frame.tobytes())
                frame_index += 1
                if frame_index % 100 == 0 or frame_index == total_frames:
                    print(f"Rendered {frame_index:,}/{total_frames:,} frames", flush=True)
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg encoder failed with status {return_code}")

    summary: dict[str, object] = {
        "source_video": str(source.relative_to(PROJECT_ROOT)),
        "output_video": str(output.relative_to(PROJECT_ROOT)),
        "detector": "combined AAM + EBCA + FM + KWConv enhanced YOLO detector",
        "weights": "Results/training/aam_ebca_fm_kwconv_green/weights/best.pt",
        "frames_processed": frame_index,
        "frames_with_detections": frames_with_detections,
        "detections_rendered": detection_count,
        "source_fps": fps,
        "resolution": [width, height],
        "depth_filter_applied": False,
        "label": "apple + confidence",
        "depth_visible": False,
        "track_id_visible": False,
        "count_visible": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_VIDEO)
    parser.add_argument("--detections-csv", type=Path, default=DETECTIONS_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_VIDEO)
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="h264_nvenc")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.detections_csv, args.output, args.encoder), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
