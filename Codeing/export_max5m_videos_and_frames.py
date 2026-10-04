#!/usr/bin/env python3
"""Export depth-filtered apple videos and every annotated detection frame.

Rows whose corrected AdaBins-derived depth is greater than ``--max-depth`` are
excluded from every output. The project's depth has no camera calibration, so
the displayed ``m est.`` values are estimates rather than measured metres.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Iterator, TextIO

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO = PROJECT_ROOT / "Dataset" / "DJI_0418.MP4"
DETECTIONS_CSV = PROJECT_ROOT / "Results" / "csv" / "apple_detections_confidence_depth.csv"
TRACKS_CSV = PROJECT_ROOT / "Results" / "csv" / "apple_tracks_id_confidence_depth.csv"
VIDEO_DIR = PROJECT_ROOT / "Results" / "inference" / "max5m_estimated"
FRAME_DIR = PROJECT_ROOT / "Results" / "frames" / "max5m_estimated"


CATEGORIES = (
    "apple_confidence",
    "apple_confidence_depth",
    "apple_confidence_depth_id",
)


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


def draw_box(image, row: dict[str, str], label: str, colour: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
    x1, x2 = sorted((max(0, x1), min(image.shape[1] - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(image.shape[0] - 1, y2)))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 7, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 4, cv2.LINE_AA)

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


def prepare_directories(video_dir: Path, frame_dir: Path) -> dict[str, Path]:
    video_dir.mkdir(parents=True, exist_ok=True)
    folders = {category: frame_dir / category for category in CATEGORIES}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            raise FileExistsError(
                f"Frame folder is not empty: {folder}. Move or remove the old export before rerunning."
            )
    return folders


def export(
    source: Path,
    detections_csv: Path,
    tracks_csv: Path,
    video_dir: Path,
    frame_dir: Path,
    max_depth: float,
    encoder: str,
    jpeg_quality: int,
) -> dict[str, object]:
    for required in (source, detections_csv, tracks_csv):
        if not required.is_file():
            raise FileNotFoundError(required)
    frame_folders = prepare_directories(video_dir, frame_dir)
    video_paths = {
        category: video_dir / f"{category}_max{max_depth:g}m_estimated.mp4"
        for category in CATEGORIES
    }

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Could not open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    encoders: dict[str, subprocess.Popen[bytes]] = {}
    output_frame_counts = {category: 0 for category in CATEGORIES}
    kept_box_counts = {category: 0 for category in CATEGORIES}
    previous_track_sides: dict[int, float] = {}
    crossing_track_ids: set[int] = set()
    started = time.monotonic()

    try:
        with ExitStack() as stack:
            detection_handle = stack.enter_context(detections_csv.open(encoding="utf-8", newline=""))
            track_handle = stack.enter_context(tracks_csv.open(encoding="utf-8", newline=""))
            detections = FrameCursor(grouped_rows(detection_handle))
            tracks = FrameCursor(grouped_rows(track_handle))
            encoders = {
                category: start_encoder(path, width, height, fps, encoder)
                for category, path in video_paths.items()
            }

            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detection_rows = [
                    row for row in detections.at(frame_index)
                    if row.get("relative_depth") and float(row["relative_depth"]) <= max_depth
                ]
                track_rows = [
                    row for row in tracks.at(frame_index)
                    if row.get("relative_depth") and float(row["relative_depth"]) <= max_depth
                ]
                images = {category: frame.copy() for category in CATEGORIES}

                for row in detection_rows:
                    confidence = float(row["confidence"])
                    depth = float(row["relative_depth"])
                    draw_box(images["apple_confidence"], row, f"apple C:{confidence:.2f}", (0, 255, 255))
                    draw_box(
                        images["apple_confidence_depth"],
                        row,
                        f"apple C:{confidence:.2f} D:{depth:.2f}m est.",
                        (40, 255, 40),
                    )
                for row in track_rows:
                    confidence = float(row["confidence"])
                    depth = float(row["relative_depth"])
                    track_id = int(row["track_id"])
                    side = (float(row["y1"]) + float(row["y2"])) / 2 - height / 2
                    previous_side = previous_track_sides.get(track_id)
                    if previous_side is not None and previous_side * side < 0:
                        crossing_track_ids.add(track_id)
                    previous_track_sides[track_id] = side
                    draw_box(
                        images["apple_confidence_depth_id"],
                        row,
                        f"apple C:{confidence:.2f} D:{depth:.2f}m est. ID:{track_id}",
                        (255, 220, 0),
                    )

                has_rows = {
                    "apple_confidence": bool(detection_rows),
                    "apple_confidence_depth": bool(detection_rows),
                    "apple_confidence_depth_id": bool(track_rows),
                }
                kept_box_counts["apple_confidence"] += len(detection_rows)
                kept_box_counts["apple_confidence_depth"] += len(detection_rows)
                kept_box_counts["apple_confidence_depth_id"] += len(track_rows)
                for category, image in images.items():
                    pipe = encoders[category].stdin
                    if pipe is None:
                        raise RuntimeError(f"Encoder stdin closed unexpectedly for {category}")
                    pipe.write(image.tobytes())
                    if has_rows[category]:
                        frame_path = frame_folders[category] / f"frame_{frame_index:06d}.jpg"
                        if not cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                            raise OSError(f"Could not write {frame_path}")
                        output_frame_counts[category] += 1

                frame_index += 1
                if frame_index % 100 == 0 or frame_index == total_frames:
                    elapsed = time.monotonic() - started
                    print(
                        f"Processed {frame_index:,}/{total_frames:,} frames "
                        f"({100 * frame_index / max(total_frames, 1):.1f}%) in {elapsed:.1f}s",
                        flush=True,
                    )
    finally:
        capture.release()
        failures = []
        for category, process in encoders.items():
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code:
                failures.append(f"{category}={return_code}")
        if failures:
            raise RuntimeError("ffmpeg encoder failure: " + ", ".join(failures))

    summary: dict[str, object] = {
        "source_video": str(source.relative_to(PROJECT_ROOT)),
        "frames_processed": frame_index,
        "source_fps": fps,
        "resolution": [width, height],
        "filter": f"relative_depth <= {max_depth:g}",
        "display_unit": "m est.",
        "depth_warning": (
            "Estimated monocular AdaBins-derived depth with apparent-scale correction; "
            "not camera-calibrated or verified physical metres."
        ),
        "kept_boxes": kept_box_counts,
        "saved_annotated_frames": output_frame_counts,
        "filtered_unique_line_crossing_count": len(crossing_track_ids),
        "count_line_y": height / 2,
        "videos": {category: str(path.relative_to(PROJECT_ROOT)) for category, path in video_paths.items()},
        "frame_folders": {
            category: str(path.relative_to(PROJECT_ROOT)) for category, path in frame_folders.items()
        },
        "labels": {
            "apple_confidence": "apple + confidence",
            "apple_confidence_depth": "apple + confidence + estimated depth",
            "apple_confidence_depth_id": "apple + confidence + estimated depth + track ID",
        },
        "count_overlay": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    summary_path = video_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_VIDEO)
    parser.add_argument("--detections-csv", type=Path, default=DETECTIONS_CSV)
    parser.add_argument("--tracks-csv", type=Path, default=TRACKS_CSV)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--frame-dir", type=Path, default=FRAME_DIR)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="h264_nvenc")
    parser.add_argument("--jpeg-quality", type=int, choices=range(70, 101), default=90, metavar="70-100")
    args = parser.parse_args()
    summary = export(
        args.source,
        args.detections_csv,
        args.tracks_csv,
        args.video_dir,
        args.frame_dir,
        args.max_depth,
        args.encoder,
        args.jpeg_quality,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
