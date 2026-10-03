#!/usr/bin/env python3
"""Re-render the three final apple-result videos from the published CSV tables.

The renderer intentionally shows only the requested shorthand labels (ID, C,
and D), uses bright outlined boxes, and does not draw a count or counting line.
The corrected D values are relative depths: lower means nearer and higher means
farther; they are not calibrated metres.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Iterator, TextIO

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO = PROJECT_ROOT / "Dataset" / "DJI_0418.MP4"
DETECTIONS_CSV = PROJECT_ROOT / "Results" / "csv" / "apple_detections_confidence_depth.csv"
TRACKS_CSV = PROJECT_ROOT / "Results" / "csv" / "apple_tracks_id_confidence_depth.csv"
OUTPUT_DIR = PROJECT_ROOT / "Results" / "inference"


def frame_rows(handle: TextIO) -> Iterator[tuple[int, list[dict[str, str]]]]:
    """Yield consecutive CSV rows grouped by their integer frame number."""
    rows = csv.DictReader(handle)
    for frame, grouped in groupby(rows, key=lambda row: int(row["frame"])):
        yield frame, list(grouped)


class FrameCursor:
    """Advance a grouped CSV iterator in lockstep with video frames."""

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


def start_encoder(path: Path, width: int, height: int, fps: float, encoder: str) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.12g}", "-i", "-", "-an", "-c:v", encoder,
    ]
    if encoder == "libx264":
        command += ["-preset", "medium", "-crf", "19"]
    elif encoder == "h264_nvenc":
        command += ["-preset", "p4", "-cq", "19"]
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def draw_box(image, row: dict[str, str], label: str, colour: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 7, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 4, cv2.LINE_AA)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    text_y = max(text_height + baseline + 4, y1 - 7)
    cv2.rectangle(
        image,
        (max(0, x1 - 2), max(0, text_y - text_height - baseline - 4)),
        (min(image.shape[1] - 1, x1 + text_width + 5), min(image.shape[0] - 1, text_y + baseline + 2)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, label, (x1 + 1, text_y), font, scale, colour, thickness, cv2.LINE_AA)


def render(source: Path, detections_csv: Path, tracks_csv: Path, output_dir: Path, encoder: str) -> None:
    for required in (source, detections_csv, tracks_csv):
        if not required.is_file():
            raise FileNotFoundError(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Could not open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    outputs = {
        "id_depth": output_dir / "apples_id_depth.mp4",
        "confidence_depth": output_dir / "apples_confidence_depth.mp4",
        "id_confidence": output_dir / "apples_id_confidence.mp4",
    }

    encoders: dict[str, subprocess.Popen[bytes]] = {}
    try:
        with ExitStack() as stack:
            detection_handle = stack.enter_context(detections_csv.open(encoding="utf-8", newline=""))
            track_handle = stack.enter_context(tracks_csv.open(encoding="utf-8", newline=""))
            detections = FrameCursor(frame_rows(detection_handle))
            tracks = FrameCursor(frame_rows(track_handle))
            encoders = {name: start_encoder(path, width, height, fps, encoder) for name, path in outputs.items()}

            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                id_depth, confidence_depth, id_confidence = frame.copy(), frame.copy(), frame.copy()
                for row in detections.at(frame_index):
                    label = f"C:{float(row['confidence']):.2f} D:{float(row['relative_depth']):.2f}"
                    draw_box(confidence_depth, row, label, (0, 255, 255))
                for row in tracks.at(frame_index):
                    draw_box(id_depth, row, f"ID:{row['track_id']} D:{float(row['relative_depth']):.2f}", (40, 255, 40))
                    draw_box(id_confidence, row, f"ID:{row['track_id']} C:{float(row['confidence']):.2f}", (255, 220, 0))
                for name, image in (
                    ("id_depth", id_depth),
                    ("confidence_depth", confidence_depth),
                    ("id_confidence", id_confidence),
                ):
                    pipe = encoders[name].stdin
                    if pipe is None:
                        raise RuntimeError(f"Encoder stdin closed unexpectedly for {name}")
                    pipe.write(image.tobytes())
                frame_index += 1
                if frame_index % 100 == 0 or frame_index == total:
                    print(f"Rendered {frame_index:,}/{total:,} frames", flush=True)
    finally:
        capture.release()
        failures = []
        for name, process in encoders.items():
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code:
                failures.append(f"{name}={return_code}")
        if failures:
            raise RuntimeError("ffmpeg encoder failure: " + ", ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_VIDEO)
    parser.add_argument("--detections-csv", type=Path, default=DETECTIONS_CSV)
    parser.add_argument("--tracks-csv", type=Path, default=TRACKS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264")
    args = parser.parse_args()
    render(args.source, args.detections_csv, args.tracks_csv, args.output_dir, args.encoder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
