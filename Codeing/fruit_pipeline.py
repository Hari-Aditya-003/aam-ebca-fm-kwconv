"""Local orchard-video preparation, training, and depth-aware tracking helpers.

All generated annotations and augmented images live under ``Dataset/``. All
trained-model artifacts, validation reports, and annotated videos live under
``Results/``. Neither directory is committed to Git.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


CLASS_NAMES = ("green_apple", "yellow_apple")
GREEN_APPLE = 0
YELLOW_APPLE = 1


class VideoUnavailableError(RuntimeError):
    """Raised when the input video cannot be decoded."""


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical on-disk layout for this research project."""

    root: Path

    @property
    def dataset(self) -> Path:
        return self.root / "Dataset"

    @property
    def results(self) -> Path:
        return self.root / "Results"

    @property
    def raw_video(self) -> Path:
        return self.dataset / "DJI_0418.MP4"

    @property
    def frames(self) -> Path:
        return self.dataset / "frames" / "raw"

    @property
    def seed_labels(self) -> Path:
        return self.dataset / "labels" / "seed_candidates"

    @property
    def auto_labels(self) -> Path:
        """Pseudo-labels created by the automatic apple detector."""
        return self.dataset / "labels" / "auto"

    @property
    def verified_labels(self) -> Path:
        return self.dataset / "labels" / "verified"

    @property
    def seed_previews(self) -> Path:
        return self.dataset / "labels" / "seed_previews"

    @property
    def auto_previews(self) -> Path:
        return self.dataset / "labels" / "auto_previews"

    @property
    def manifests(self) -> Path:
        return self.dataset / "manifests"

    @property
    def augmented_images(self) -> Path:
        return self.dataset / "augmentations" / "images" / "train"

    @property
    def augmented_labels(self) -> Path:
        return self.dataset / "augmentations" / "labels" / "train"

    @property
    def yolo(self) -> Path:
        return self.dataset / "yolo"

    @property
    def yolo_images(self) -> Path:
        return self.yolo / "images"

    @property
    def yolo_labels(self) -> Path:
        return self.yolo / "labels"

    @property
    def yolo_yaml(self) -> Path:
        return self.yolo / "apple_dataset.yaml"

    @property
    def training_results(self) -> Path:
        return self.results / "training"

    @property
    def inference_results(self) -> Path:
        return self.results / "inference"

    def ensure_layout(self) -> None:
        """Create output folders without changing the source video."""
        for folder in (
            self.dataset,
            self.results,
            self.frames,
            self.seed_labels,
            self.auto_labels,
            self.verified_labels,
            self.seed_previews,
            self.auto_previews,
            self.manifests,
            self.augmented_images,
            self.augmented_labels,
            self.training_results,
            self.inference_results,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (self.yolo_images / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_labels / split).mkdir(parents=True, exist_ok=True)


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the project root even when the notebook runs from ``Codeing/``."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "Dataset").exists() and (candidate / "Results").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the project root. Start the notebook from the repository root or Codeing/."
    )


def project_paths(start: str | Path | None = None) -> ProjectPaths:
    paths = ProjectPaths(find_project_root(start))
    paths.ensure_layout()
    return paths


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def video_preflight(video_path: str | Path) -> dict[str, Any]:
    """Return an explicit decoder status without mutating the input video."""
    path = Path(video_path)
    report: dict[str, Any] = {"path": str(path), "exists": path.exists(), "playable": False}
    if not path.exists():
        report["error"] = "File does not exist."
        return report

    capture = cv2.VideoCapture(str(path))
    report.update(
        {
            "opened": bool(capture.isOpened()),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    )
    readable, _ = capture.read() if capture.isOpened() else (False, None)
    capture.release()
    report["playable"] = bool(report["opened"] and readable)
    if not report["playable"]:
        report["error"] = (
            "OpenCV cannot decode a frame. Re-export the source clip to a standard H.264 MP4 "
            "or repair it before running extraction."
        )
    return report


def require_playable_video(video_path: str | Path) -> dict[str, Any]:
    report = video_preflight(video_path)
    if not report["playable"]:
        raise VideoUnavailableError(report.get("error", "Input video is not playable."))
    return report


def extract_uniform_frames(
    paths: ProjectPaths,
    *,
    video_path: str | Path | None = None,
    sample_fps: float = 5.0,
    max_frames: int = 800,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Extract temporally uniform frames for annotation.

    ``max_frames`` prevents a long video from producing an unmanageable manual
    labelling task. The source-frame index and timestamp are persisted in a
    manifest so train/validation/test splits remain temporally disjoint.
    """
    if sample_fps <= 0 or max_frames < 1:
        raise ValueError("sample_fps and max_frames must be positive")
    input_video = Path(video_path or paths.raw_video)
    info = require_playable_video(input_video)
    source_fps = info["fps"]
    total_frames = info["frame_count"]
    if source_fps <= 0 or total_frames <= 0:
        raise VideoUnavailableError("The input video has no usable frame-rate or frame-count metadata.")

    requested = min(max_frames, max(1, math.ceil(total_frames / source_fps * sample_fps)))
    indices = np.unique(np.linspace(0, total_frames - 1, requested, dtype=np.int64))
    if overwrite:
        # ``frames/raw`` contains generated frame extractions only. Clearing the
        # old names avoids mixing separate sampling runs (for example a previous
        # 5 FPS run with a new exact-800 run) in one training dataset.
        for existing_frame in paths.frames.glob("frame_*_src_*.jpg"):
            existing_frame.unlink()
    capture = cv2.VideoCapture(str(input_video))
    manifest: list[dict[str, Any]] = []
    try:
        for ordinal, source_index in enumerate(indices):
            target = paths.frames / f"frame_{ordinal:05d}_src_{source_index:08d}.jpg"
            if target.exists() and not overwrite:
                manifest.append(
                    {
                        "image": target.name,
                        "source_frame": int(source_index),
                        "timestamp_s": round(float(source_index) / source_fps, 6),
                    }
                )
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
            ok, frame = capture.read()
            if not ok:
                continue
            if not cv2.imwrite(str(target), frame):
                raise OSError(f"Could not write extracted frame: {target}")
            manifest.append(
                {
                    "image": target.name,
                    "source_frame": int(source_index),
                    "timestamp_s": round(float(source_index) / source_fps, 6),
                }
            )
    finally:
        capture.release()

    _write_jsonl(paths.manifests / "raw_frames.jsonl", manifest)
    return manifest


@dataclass(frozen=True)
class YoloBox:
    """One normalized YOLO detection box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"class_id must be in [0, {len(CLASS_NAMES) - 1}]")
        for value in (self.x_center, self.y_center, self.width, self.height):
            if not 0.0 <= value <= 1.0:
                raise ValueError("YOLO coordinates must lie in [0, 1]")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("YOLO boxes need positive width and height")

    def as_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.7f} {self.y_center:.7f} "
            f"{self.width:.7f} {self.height:.7f}"
        )

    def xyxy(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x1 = round((self.x_center - self.width / 2) * image_width)
        y1 = round((self.y_center - self.height / 2) * image_height)
        x2 = round((self.x_center + self.width / 2) * image_width)
        y2 = round((self.y_center + self.height / 2) * image_height)
        return x1, y1, x2, y2


def xyxy_to_yolo(
    class_id: int, xyxy: Sequence[float], image_width: int, image_height: int
) -> YoloBox:
    x1, y1, x2, y2 = xyxy
    x1, x2 = np.clip([x1, x2], 0, image_width)
    y1, y2 = np.clip([y1, y2], 0, image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Cannot create a label from an empty box")
    return YoloBox(
        class_id=int(class_id),
        x_center=float((x1 + x2) / (2 * image_width)),
        y_center=float((y1 + y2) / (2 * image_height)),
        width=float((x2 - x1) / image_width),
        height=float((y2 - y1) / image_height),
    )


def read_yolo_labels(path: str | Path) -> list[YoloBox]:
    label_path = Path(path)
    if not label_path.exists():
        return []
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Malformed YOLO label in {label_path}:{line_number}")
        boxes.append(YoloBox(int(values[0]), *(float(value) for value in values[1:])))
    return boxes


def write_yolo_labels(path: str | Path, boxes: Iterable[YoloBox]) -> None:
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(box.as_line() for box in boxes) + "\n", encoding="utf-8")


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _nms(candidates: list[tuple[int, tuple[int, int, int, int], float]], threshold: float = 0.45) -> list[
    tuple[int, tuple[int, int, int, int], float]
]:
    kept: list[tuple[int, tuple[int, int, int, int], float]] = []
    for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
        if all(_box_iou(candidate[1], existing[1]) < threshold for existing in kept):
            kept.append(candidate)
    return kept


def _candidate_boxes_for_mask(mask: np.ndarray, class_id: int, min_area: int) -> list[
    tuple[int, tuple[int, int, int, int], float]
]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = min(width, height) / max(width, height)
        fill_ratio = area / max(1.0, width * height)
        if aspect_ratio < 0.45 or fill_ratio < 0.30 or circularity < 0.30:
            continue
        score = 0.55 * circularity + 0.25 * aspect_ratio + 0.20 * min(fill_ratio, 1.0)
        candidates.append((class_id, (x, y, x + width, y + height), score))
    return candidates


def colour_seed_boxes(image: np.ndarray, min_area: int = 120) -> list[YoloBox]:
    """Create *review-only* green/yellow apple candidates from HSV colour masks.

    Green foliage has a similar HSV range to green apples, so these candidates
    are never used as ground truth until a person reviews them.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = {
        GREEN_APPLE: ((35, 40, 35), (95, 255, 255)),
        YELLOW_APPLE: ((18, 70, 70), (34, 255, 255)),
    }
    kernel = np.ones((5, 5), dtype=np.uint8)
    candidates: list[tuple[int, tuple[int, int, int, int], float]] = []
    for class_id, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        candidates.extend(_candidate_boxes_for_mask(mask, class_id, min_area=min_area))
    height, width = image.shape[:2]
    return [xyxy_to_yolo(class_id, xyxy, width, height) for class_id, xyxy, _ in _nms(candidates)]


def draw_yolo_boxes(image: np.ndarray, boxes: Iterable[YoloBox], *, include_score: bool = False) -> np.ndarray:
    """Return a preview image; green and yellow labels use matching colours."""
    preview = image.copy()
    colours = {GREEN_APPLE: (40, 180, 40), YELLOW_APPLE: (0, 220, 240)}
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy(image.shape[1], image.shape[0])
        colour = colours[box.class_id]
        cv2.rectangle(preview, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            preview,
            CLASS_NAMES[box.class_id],
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    return preview


def create_colour_seed_labels(
    paths: ProjectPaths, *, image_paths: Iterable[Path] | None = None, overwrite: bool = False
) -> dict[str, int]:
    """Save review candidates and overlays under ``Dataset/labels``."""
    images = list(image_paths or sorted(paths.frames.glob("*.jpg")))
    rows: list[dict[str, Any]] = []
    total_boxes = 0
    for image_path in images:
        label_path = paths.seed_labels / f"{image_path.stem}.txt"
        preview_path = paths.seed_previews / image_path.name
        if label_path.exists() and preview_path.exists() and not overwrite:
            boxes = read_yolo_labels(label_path)
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            boxes = colour_seed_boxes(image)
            write_yolo_labels(label_path, boxes)
            cv2.imwrite(str(preview_path), draw_yolo_boxes(image, boxes))
        total_boxes += len(boxes)
        rows.append({"image": image_path.name, "candidate_boxes": len(boxes), "status": "needs_review"})
    _write_jsonl(paths.manifests / "seed_candidates.jsonl", rows)
    return {"images": len(rows), "candidate_boxes": total_boxes}


def review_frame_with_opencv(paths: ProjectPaths, image_path: str | Path) -> list[YoloBox]:
    """Manually draw verified boxes and save them as YOLO labels.

    This local-desktop helper deliberately replaces the colour candidates rather
    than treating them as truth. Press Enter after drawing all ROIs, then type
    ``g`` or ``y`` for each box in the notebook console.
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    window_name = "Draw all apple boxes, then press Enter"
    rois = cv2.selectROIs(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    boxes: list[YoloBox] = []
    for x, y, width, height in rois:
        while True:
            answer = input(f"Class for ROI {(x, y, width, height)} [g=green, y=yellow, s=skip]: ").strip().lower()
            if answer in {"g", "green"}:
                boxes.append(xyxy_to_yolo(GREEN_APPLE, (x, y, x + width, y + height), image.shape[1], image.shape[0]))
                break
            if answer in {"y", "yellow"}:
                boxes.append(xyxy_to_yolo(YELLOW_APPLE, (x, y, x + width, y + height), image.shape[1], image.shape[0]))
                break
            if answer in {"s", "skip"}:
                break
            print("Enter g, y, or s.")
    write_yolo_labels(paths.verified_labels / f"{path.stem}.txt", boxes)
    return boxes


def _split_counts(total: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("At least three verified frames are needed for train/val/test splits")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must form a valid split")
    train = max(1, int(total * train_fraction))
    val = max(1, int(total * val_fraction))
    test = total - train - val
    if test < 1:
        test = 1
        train = max(1, train - 1)
    return train, val, test


def _copy_pair(image_path: Path, source_label: Path, image_target: Path, label_target: Path) -> None:
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_target / image_path.name)
    # An empty label file is valid only after a reviewer explicitly decided this
    # frame contains no apples.
    shutil.copy2(source_label, label_target / source_label.name)


def build_temporal_yolo_dataset(
    paths: ProjectPaths,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    label_directory: str | Path | None = None,
) -> dict[str, int]:
    """Copy one label set into temporally disjoint YOLO splits.

    The split happens *before* augmentation. This prevents adjacent video frames
    or an augmented copy of a validation/test frame leaking into training.

    ``label_directory`` may be the manually verified labels folder or the
    automatic pseudo-labels folder. Every raw frame must have a label file;
    empty files deliberately encode automatic negative detections.
    """
    images = sorted(paths.frames.glob("*.jpg"))
    source_labels = Path(label_directory) if label_directory is not None else paths.verified_labels
    pairs = []
    missing_labels = []
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        # Parsing here catches malformed labels before training begins.
        read_yolo_labels(label_path)
        pairs.append((image_path, label_path))
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} images have no label file in {source_labels}. "
            "Run automatic labelling again so each image has a label file (possibly empty)."
        )
    train_count, val_count, _ = _split_counts(len(pairs), train_fraction, val_fraction)
    split_pairs = {
        "train": pairs[:train_count],
        "val": pairs[train_count : train_count + val_count],
        "test": pairs[train_count + val_count :],
    }
    manifest = []
    for split, items in split_pairs.items():
        for image_path, label_path in items:
            _copy_pair(image_path, label_path, paths.yolo_images / split, paths.yolo_labels / split)
            manifest.append(
                {
                    "image": image_path.name,
                    "split": split,
                    "label": label_path.name,
                    "label_directory": str(source_labels),
                }
            )
    _write_jsonl(paths.manifests / "temporal_split.jsonl", manifest)
    _write_dataset_yaml(paths)
    return {split: len(items) for split, items in split_pairs.items()}


def _write_dataset_yaml(paths: ProjectPaths) -> None:
    root_as_yaml = json.dumps(str(paths.yolo.resolve()))
    paths.yolo_yaml.write_text(
        "\n".join(
            (
                f"path: {root_as_yaml}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: green_apple",
                "  1: yellow_apple",
                "",
            )
        ),
        encoding="utf-8",
    )


def _flip_horizontal(boxes: Iterable[YoloBox]) -> list[YoloBox]:
    return [YoloBox(box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height) for box in boxes]


def _brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _motion_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    angle = random.uniform(0, 180)
    matrix = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def _leaf_occlusion(image: np.ndarray, boxes: Sequence[YoloBox], rng: random.Random) -> np.ndarray:
    """Overlay a small foliage-coloured patch to train partial-occlusion robustness."""
    output = image.copy()
    height, width = output.shape[:2]
    if boxes:
        target = rng.choice(list(boxes))
        x1, y1, x2, y2 = target.xyxy(width, height)
        patch_width = max(3, int((x2 - x1) * rng.uniform(0.18, 0.35)))
        patch_height = max(3, int((y2 - y1) * rng.uniform(0.18, 0.35)))
        left = rng.randint(x1, max(x1, x2 - patch_width))
        top = rng.randint(y1, max(y1, y2 - patch_height))
    else:
        patch_width, patch_height = max(8, width // 20), max(8, height // 20)
        left = rng.randint(0, max(0, width - patch_width))
        top = rng.randint(0, max(0, height - patch_height))
    colour = (rng.randint(15, 70), rng.randint(55, 140), rng.randint(10, 65))
    overlay = output.copy()
    cv2.ellipse(
        overlay,
        (left + patch_width // 2, top + patch_height // 2),
        (patch_width // 2, patch_height // 2),
        rng.randint(0, 179),
        0,
        360,
        colour,
        -1,
    )
    return cv2.addWeighted(overlay, 0.70, output, 0.30, 0)


def augment_training_split(
    paths: ProjectPaths, *, seed: int = 42, overwrite: bool = False
) -> dict[str, int]:
    """Create label-safe, colour-preserving train-only augmentations.

    Strong hue shifts are intentionally excluded because they could turn a green
    apple into a yellow class (or vice versa). Each result is saved both in the
    explicit augmentation folder and in YOLO's training split.
    """
    rng = random.Random(seed)
    source_images = [path for path in sorted((paths.yolo_images / "train").glob("*.jpg")) if "__aug_" not in path.stem]
    created = {"horizontal_flip": 0, "dark": 0, "bright": 0, "motion_blur": 0, "leaf_occlusion": 0}
    for image_path in source_images:
        labels = read_yolo_labels(paths.yolo_labels / "train" / f"{image_path.stem}.txt")
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        variants = {
            "horizontal_flip": (cv2.flip(image, 1), _flip_horizontal(labels)),
            "dark": (_brightness(image, 0.70), labels),
            "bright": (_brightness(image, 1.25), labels),
            "motion_blur": (_motion_blur(image), labels),
            "leaf_occlusion": (_leaf_occlusion(image, labels, rng), labels),
        }
        for name, (variant, variant_labels) in variants.items():
            output_stem = f"{image_path.stem}__aug_{name}"
            image_output = paths.augmented_images / f"{output_stem}.jpg"
            label_output = paths.augmented_labels / f"{output_stem}.txt"
            if not image_output.exists() or overwrite:
                if not cv2.imwrite(str(image_output), variant):
                    raise OSError(f"Could not write augmented image: {image_output}")
                write_yolo_labels(label_output, variant_labels)
            _copy_pair(
                image_output,
                label_output,
                paths.yolo_images / "train",
                paths.yolo_labels / "train",
            )
            created[name] += 1
    _write_json(paths.manifests / "augmentation_summary.json", created)
    return created


def validate_yolo_dataset(paths: ProjectPaths) -> dict[str, Any]:
    """Check image/label parity and report green/yellow class balance."""
    report: dict[str, Any] = {"splits": {}, "classes": {name: 0 for name in CLASS_NAMES}}
    for split in ("train", "val", "test"):
        images = sorted((paths.yolo_images / split).glob("*.jpg"))
        missing_labels = []
        box_count = 0
        for image_path in images:
            label_path = paths.yolo_labels / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue
            labels = read_yolo_labels(label_path)
            box_count += len(labels)
            for label in labels:
                report["classes"][CLASS_NAMES[label.class_id]] += 1
        report["splits"][split] = {
            "images": len(images),
            "boxes": box_count,
            "missing_labels": missing_labels,
        }
    report["valid"] = all(not split["missing_labels"] for split in report["splits"].values())
    _write_json(paths.manifests / "dataset_validation.json", report)
    return report


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ModuleNotFoundError as error:
        raise RuntimeError("Install torch before creating AAM/EBCA/FM/KWConv modules.") from error
    return torch, nn, functional


try:
    torch, nn, F = _require_torch()
except RuntimeError:
    torch = nn = F = None


if nn is not None:

    class AAM(nn.Module):
        """Attribute attention: colour, shape, and scale branches with learned fusion."""

        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            hidden = max(1, channels // reduction)
            self.colour = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1)
            )
            self.shape = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            self.scale = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), nn.Conv2d(channels, channels, 1))
            self.branch_weights = nn.Parameter(torch.zeros(3))

        def forward(self, features: Any) -> Any:
            colour = self.colour(features).expand_as(features)
            shape = self.shape(features)
            scale = self.scale(features)
            alpha = torch.softmax(self.branch_weights, dim=0)
            attention = torch.sigmoid(alpha[0] * colour + alpha[1] * shape + alpha[2] * scale)
            return features * attention


    class EBCA(nn.Module):
        """Memory-efficient bidirectional cross-attention between P3 and P4."""

        def __init__(self, p3_channels: int, p4_channels: int) -> None:
            super().__init__()
            self.p4_to_p3 = nn.Conv2d(p4_channels, p3_channels, 1, bias=False)
            self.p3_to_p4 = nn.Conv2d(p3_channels, p4_channels, 1, bias=False)
            self.p3_query = nn.Conv2d(p3_channels, p3_channels, 1, bias=False)
            self.p4_query = nn.Conv2d(p4_channels, p4_channels, 1, bias=False)
            self.p3_norm = nn.BatchNorm2d(p3_channels)
            self.p4_norm = nn.BatchNorm2d(p4_channels)

        def forward(self, p3: Any, p4: Any) -> tuple[Any, Any]:
            p4_up = F.interpolate(self.p4_to_p3(p4), size=p3.shape[-2:], mode="bilinear", align_corners=False)
            p3_gate = torch.sigmoid(self.p3_query(p3) * p4_up / math.sqrt(p3.shape[1]))
            enhanced_p3 = self.p3_norm(p3 + p3_gate * p4_up)
            p3_down = F.adaptive_avg_pool2d(self.p3_to_p4(p3), p4.shape[-2:])
            p4_gate = torch.sigmoid(self.p4_query(p4) * p3_down / math.sqrt(p4.shape[1]))
            enhanced_p4 = self.p4_norm(p4 + p4_gate * p3_down)
            return enhanced_p3, enhanced_p4


    class FM(nn.Module):
        """Focal modulation with hierarchical local context for occluded fruit."""

        def __init__(self, channels: int, levels: int = 3) -> None:
            super().__init__()
            self.query = nn.Conv2d(channels, channels, 1)
            self.context = nn.ModuleList(
                [nn.Conv2d(channels, channels, 3 + 2 * level, padding=1 + level, groups=channels) for level in range(levels)]
            )
            self.gates = nn.Conv2d(channels, levels + 1, 1)
            self.modulator = nn.Conv2d(channels, channels, 1)
            self.output = nn.Conv2d(channels, channels, 1)

        def forward(self, features: Any) -> Any:
            query = self.query(features)
            gates = torch.softmax(self.gates(features), dim=1)
            context = 0
            current = features
            for level, layer in enumerate(self.context):
                current = layer(current)
                context = context + gates[:, level : level + 1] * current
            global_context = F.adaptive_avg_pool2d(features, 1)
            context = context + gates[:, -1:] * global_context
            return self.output(query * self.modulator(context))


    class KWConv(nn.Module):
        """Input-adaptive mixture of 3×3, 5×5, and 7×7 convolution kernels."""

        def __init__(self, channels: int, kernels: Sequence[int] = (3, 5, 7)) -> None:
            super().__init__()
            self.branches = nn.ModuleList(
                [nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False) for kernel in kernels]
            )
            self.selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, len(kernels), 1))
            self.norm = nn.BatchNorm2d(channels)
            self.activation = nn.SiLU()

        def forward(self, features: Any) -> Any:
            weights = torch.softmax(self.selector(features), dim=1)
            mixed = sum(weights[:, index : index + 1] * branch(features) for index, branch in enumerate(self.branches))
            return self.activation(self.norm(mixed))


    class ProposedEnhancementNeck(nn.Module):
        """AAM → EBCA → FM with KWConv at each feature scale (P3/P4/P5)."""

        def __init__(self, p3_channels: int, p4_channels: int, p5_channels: int) -> None:
            super().__init__()
            self.aam = AAM(p4_channels)
            self.ebca = EBCA(p3_channels, p4_channels)
            self.fm = FM(p4_channels)
            self.kw_p3 = KWConv(p3_channels)
            self.kw_p4 = KWConv(p4_channels)
            self.kw_p5 = KWConv(p5_channels)

        def forward(self, p3: Any, p4: Any, p5: Any) -> tuple[Any, Any, Any]:
            p4 = self.aam(p4)
            p3, p4 = self.ebca(p3, p4)
            p4 = self.fm(p4)
            return self.kw_p3(p3), self.kw_p4(p4), self.kw_p5(p5)


    class EnhancedDetectHead(nn.Module):
        """Apply the proposed P3/P4/P5 enhancement directly before YOLO Detect.

        The wrapper retains the original Ultralytics ``Detect`` head and its
        learned weights.  It only replaces the three incoming FPN features with
        AAM–EBCA–FM–KWConv-enhanced versions, so detection loss and decoding
        remain the upstream implementation.
        """

        def __init__(self, detect: Any, channels: Sequence[int]) -> None:
            super().__init__()
            if len(channels) != 3:
                raise ValueError("The proposed neck requires P3, P4, and P5 channels.")
            self.detect = detect
            self.enhancement = ProposedEnhancementNeck(*channels)
            # Ultralytics' graph runner reads these metadata attributes from
            # every module in its sequential model.
            self.i = getattr(detect, "i", None)
            self.f = getattr(detect, "f", None)
            self.type = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self.np = sum(parameter.numel() for parameter in self.parameters())

        @property
        def stride(self) -> Any:
            return self.detect.stride

        @property
        def nc(self) -> int:
            return self.detect.nc

        @property
        def nl(self) -> int:
            return self.detect.nl

        @property
        def no(self) -> int:
            return self.detect.no

        @property
        def reg_max(self) -> int:
            return self.detect.reg_max

        @property
        def dfl(self) -> Any:
            return self.detect.dfl

        @property
        def max_det(self) -> int:
            return self.detect.max_det

        def forward(self, features: Sequence[Any]) -> Any:
            if len(features) != 3:
                raise ValueError("Expected three detection features: P3, P4, and P5.")
            return self.detect(list(self.enhancement(*features)))


else:

    class _TorchMissing:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class AAM(_TorchMissing):
        pass


    class EBCA(_TorchMissing):
        pass


    class FM(_TorchMissing):
        pass


    class KWConv(_TorchMissing):
        pass


    class ProposedEnhancementNeck(_TorchMissing):
        pass


    class EnhancedDetectHead(_TorchMissing):
        pass


def module_smoke_test(paths: ProjectPaths, device: str = "cpu") -> dict[str, Any]:
    """Validate the proposed module tensor interfaces before Ultralytics integration."""
    if torch is None:
        _require_torch()
    model = ProposedEnhancementNeck(96, 192, 384).to(device).eval()
    with torch.inference_mode():
        outputs = model(
            torch.zeros(1, 96, 80, 80, device=device),
            torch.zeros(1, 192, 40, 40, device=device),
            torch.zeros(1, 384, 20, 20, device=device),
        )
    report = {"output_shapes": [list(output.shape) for output in outputs], "device": device}
    _write_json(paths.results / "architecture_smoke_test.json", report)
    return report


def patch_yolov8_detection_head(yolo: Any) -> Any:
    """Insert AAM–EBCA–FM–KWConv immediately before a YOLOv8 Detect head.

    ``yolo`` can be an Ultralytics ``YOLO`` object or its underlying detection
    model. The function supports standard three-scale YOLOv8 models and keeps
    the pretrained backbone/Detect weights intact.
    """
    if torch is None:
        _require_torch()
    # ``YOLO.model`` is the DetectionModel, while ``DetectionModel.model`` is
    # the graph's sequential module list. Distinguish the two nesting levels so
    # callers may pass either public Ultralytics object.
    candidate_model = getattr(yolo, "model", None)
    detection_model = candidate_model if candidate_model is not None and hasattr(candidate_model, "model") else yolo
    modules = getattr(detection_model, "model", None)
    if modules is None or len(modules) == 0:
        raise TypeError("Expected an Ultralytics YOLO detection model with a module list.")
    head = modules[-1]
    if isinstance(head, EnhancedDetectHead):
        return yolo
    try:
        channels = tuple(int(branch[0].conv.in_channels) for branch in head.cv2)
    except (AttributeError, IndexError, TypeError) as error:
        raise TypeError("Expected a standard three-scale Ultralytics Detect head.") from error
    if len(channels) != 3:
        raise ValueError(f"Expected three YOLO detection scales; found {len(channels)}.")
    modules[-1] = EnhancedDetectHead(head, channels)
    return yolo


class AdaBinsDepth:
    """Adapter for the official AdaBins ``InferenceHelper`` interface.

    Orchard RGB video has no calibrated depth ground truth. Unless the camera is
    calibrated against a metric depth source, use these predictions only for
    relative foreground/background gating, not metric distance claims.
    """

    def __init__(
        self, repository: str | Path, dataset: str = "kitti", device: str | None = None
    ) -> None:
        repository_path = Path(repository).expanduser().resolve()
        if not (repository_path / "infer.py").exists():
            raise FileNotFoundError(
                "AdaBins repository was not found. Clone the official repository and pass its directory."
            )
        if str(repository_path) not in sys.path:
            sys.path.insert(0, str(repository_path))
        try:
            from infer import InferenceHelper
        except ModuleNotFoundError as error:
            raise RuntimeError("AdaBins dependencies are missing in the current notebook kernel.") from error
        # The upstream helper resolves pretrained weights relative to the
        # current directory and defaults to CUDA. Keep those legacy assumptions
        # contained here so the notebook works on this CPU-only workstation.
        selected_device = device or ("cuda:0" if torch is not None and torch.cuda.is_available() else "cpu")
        previous_directory = Path.cwd()
        try:
            os.chdir(repository_path)
            self._helper = InferenceHelper(dataset=dataset, device=selected_device)
        finally:
            os.chdir(previous_directory)

    def predict(self, bgr_frame: np.ndarray) -> np.ndarray:
        if torch is None:
            _require_torch()
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # Call the model helper with a tensor directly. Its legacy PIL adapter
        # uses ``np.array(..., copy=False)`` which is incompatible with NumPy 2.
        image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        _, depth = self._helper.predict(image.to(self._helper.device))
        depth_map = np.asarray(depth).squeeze().astype(np.float32)
        return cv2.resize(depth_map, (bgr_frame.shape[1], bgr_frame.shape[0]), interpolation=cv2.INTER_LINEAR)


def robust_instance_depth(depth_map: np.ndarray | None, bbox: Sequence[float], trim_fraction: float = 0.10) -> float | None:
    """Use the trimmed median in the central fruit region to avoid background pixels."""
    if depth_map is None:
        return None
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    margin_x, margin_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    left, right = max(0, int(x1 + margin_x)), min(width, int(x2 - margin_x))
    top, bottom = max(0, int(y1 + margin_y)), min(height, int(y2 - margin_y))
    if right <= left or bottom <= top:
        return None
    values = depth_map[top:bottom, left:right].reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return None
    values.sort()
    trim = int(values.size * trim_fraction)
    if trim and values.size > 2 * trim:
        values = values[trim:-trim]
    return float(np.median(values))


@dataclass(frozen=True)
class FruitDetection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    depth: float | None = None


@dataclass
class FruitTrack:
    track_id: int
    bbox: np.ndarray
    class_id: int
    score: float
    depth: float | None
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    updated: bool = True

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2], dtype=np.float32)

    def predict(self) -> None:
        self.bbox[[0, 2]] += self.velocity[0]
        self.bbox[[1, 3]] += self.velocity[1]
        self.age += 1
        self.time_since_update += 1
        self.updated = False

    def update(self, detection: FruitDetection) -> None:
        previous_center = self.center
        self.bbox = np.asarray(detection.bbox, dtype=np.float32)
        self.velocity = 0.65 * self.velocity + 0.35 * (self.center - previous_center)
        self.score = detection.score
        self.depth = detection.depth if detection.depth is not None else self.depth
        self.hits += 1
        self.time_since_update = 0
        self.updated = True


class DepthAwareByteTracker:
    """Two-stage ByteTrack-style association with a depth compatibility gate."""

    def __init__(
        self,
        *,
        high_confidence: float = 0.50,
        low_confidence: float = 0.10,
        match_iou: float = 0.10,
        max_cost: float = 0.70,
        depth_relative_gate: float = 0.40,
        track_buffer: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.match_iou = match_iou
        self.max_cost = max_cost
        self.depth_relative_gate = depth_relative_gate
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self._tracks: list[FruitTrack] = []
        self._next_id = 1

    @staticmethod
    def _cost(track: FruitTrack, detection: FruitDetection, min_iou: float, depth_gate: float) -> float:
        if track.class_id != detection.class_id:
            return float("inf")
        iou = _box_iou(track.bbox, detection.bbox)
        if iou < min_iou:
            return float("inf")
        depth_cost = 0.0
        if track.depth is not None and detection.depth is not None:
            denominator = max(abs(track.depth), abs(detection.depth), 1e-6)
            relative_difference = abs(track.depth - detection.depth) / denominator
            if relative_difference > depth_gate:
                return float("inf")
            depth_cost = min(1.0, relative_difference / depth_gate)
        track_center = track.center
        detection_center = np.array(
            [(detection.bbox[0] + detection.bbox[2]) / 2, (detection.bbox[1] + detection.bbox[3]) / 2]
        )
        scale = max(1.0, math.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]))
        motion_cost = min(1.0, float(np.linalg.norm(track_center - detection_center) / (3 * scale)))
        return 0.65 * (1.0 - iou) + 0.20 * depth_cost + 0.15 * motion_cost

    def _associate(
        self, track_indices: list[int], detections: list[FruitDetection], max_cost: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not detections:
            return [], track_indices.copy(), list(range(len(detections)))
        costs = np.full((len(track_indices), len(detections)), 1e6, dtype=np.float64)
        for row, track_index in enumerate(track_indices):
            for column, detection in enumerate(detections):
                costs[row, column] = self._cost(
                    self._tracks[track_index], detection, self.match_iou, self.depth_relative_gate
                )
        try:
            from scipy.optimize import linear_sum_assignment

            rows, columns = linear_sum_assignment(costs)
        except ModuleNotFoundError:
            rows, columns = np.array([], dtype=int), np.array([], dtype=int)
            for _ in range(min(costs.shape)):
                row, column = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[row, column] > max_cost:
                    break
                rows, columns = np.append(rows, row), np.append(columns, column)
                costs[row, :] = 1e6
                costs[:, column] = 1e6
        matches: list[tuple[int, int]] = []
        used_tracks, used_detections = set(), set()
        for row, column in zip(rows, columns):
            if costs[row, column] <= max_cost:
                matches.append((track_indices[int(row)], int(column)))
                used_tracks.add(track_indices[int(row)])
                used_detections.add(int(column))
        return (
            matches,
            [index for index in track_indices if index not in used_tracks],
            [index for index in range(len(detections)) if index not in used_detections],
        )

    def update(self, detections: Iterable[FruitDetection]) -> list[FruitTrack]:
        active_detections = [detection for detection in detections if detection.score >= self.low_confidence]
        for track in self._tracks:
            track.predict()
        track_indices = list(range(len(self._tracks)))
        high = [detection for detection in active_detections if detection.score >= self.high_confidence]
        low = [detection for detection in active_detections if detection.score < self.high_confidence]

        high_matches, unmatched_tracks, unmatched_high = self._associate(track_indices, high, self.max_cost)
        for track_index, detection_index in high_matches:
            self._tracks[track_index].update(high[detection_index])
        low_matches, _, _ = self._associate(unmatched_tracks, low, min(0.85, self.max_cost + 0.10))
        for track_index, detection_index in low_matches:
            self._tracks[track_index].update(low[detection_index])
        for detection_index in unmatched_high:
            detection = high[detection_index]
            self._tracks.append(
                FruitTrack(
                    track_id=self._next_id,
                    bbox=np.asarray(detection.bbox, dtype=np.float32),
                    class_id=detection.class_id,
                    score=detection.score,
                    depth=detection.depth,
                )
            )
            self._next_id += 1
        self._tracks = [track for track in self._tracks if track.time_since_update <= self.track_buffer]
        return [track for track in self._tracks if track.updated and track.hits >= self.min_hits]


class UniqueLineCounter:
    """Count a confirmed track once when it crosses a configured virtual line."""

    def __init__(self, start: tuple[int, int], end: tuple[int, int], direction: str = "positive") -> None:
        if direction not in {"positive", "negative", "both"}:
            raise ValueError("direction must be positive, negative, or both")
        self.start, self.end, self.direction = np.array(start), np.array(end), direction
        self.counted_track_ids: set[int] = set()
        self._previous_sides: dict[int, float] = {}

    def _side(self, point: np.ndarray) -> float:
        return float(np.cross(self.end - self.start, point - self.start))

    def update(self, tracks: Iterable[FruitTrack]) -> list[int]:
        new_counts = []
        for track in tracks:
            current = self._side(track.center)
            previous = self._previous_sides.get(track.track_id)
            self._previous_sides[track.track_id] = current
            if previous is None or track.track_id in self.counted_track_ids or previous * current >= 0:
                continue
            direction_ok = self.direction == "both" or (self.direction == "positive" and current > 0) or (
                self.direction == "negative" and current < 0
            )
            if direction_ok:
                self.counted_track_ids.add(track.track_id)
                new_counts.append(track.track_id)
        return new_counts


def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("Install ultralytics before training or inference: pip install ultralytics") from error
    return YOLO


def train_baseline_yolov8(
    paths: ProjectPaths,
    *,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    image_size: int = 1024,
    batch: int | float = -1,
    device: str | int | None = None,
) -> Any:
    """Train the reproducible baseline; save all artifacts inside ``Results/training``."""
    validation = validate_yolo_dataset(paths)
    if not validation["valid"]:
        raise ValueError("Every YOLO image needs a verified label file before training.")
    YOLO = _require_ultralytics()
    model = YOLO(model_name)
    return model.train(
        data=str(paths.yolo_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(paths.training_results),
        name="yolov8_green_yellow_apple",
        exist_ok=True,
        # Colour is a class attribute, so avoid hue jitter that can swap classes.
        hsv_h=0.0,
        hsv_s=0.20,
        hsv_v=0.20,
        fliplr=0.50,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.20,
        mosaic=0.10,
        erasing=0.10,
    )


def train_proposed_yolov8(
    paths: ProjectPaths,
    *,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    image_size: int = 1024,
    batch: int | float = -1,
    device: str | int | None = None,
    run_name: str = "aam_ebca_fm_kwconv_auto",
    **trainer_overrides: Any,
) -> Any:
    """Train YOLOv8 with AAM–EBCA–FM–KWConv inserted before Detect.

    Ultralytics reconstructs a model when ``YOLO.train`` starts. A custom
    trainer patches the freshly reconstructed two-class model, rather than only
    patching the initial 80-class checkpoint in memory. This guarantees the
    proposed neck is part of the trainable model and saved checkpoint.
    """
    validation = validate_yolo_dataset(paths)
    if not validation["valid"]:
        raise ValueError("Every YOLO image needs an automatic label file before training.")
    YOLO = _require_ultralytics()
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ModuleNotFoundError as error:
        raise RuntimeError("The installed Ultralytics package does not provide DetectionTrainer.") from error

    class ProposedFruitTrainer(DetectionTrainer):
        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            patch_yolov8_detection_head(model)
            return model

    model = YOLO(model_name)
    return model.train(
        trainer=ProposedFruitTrainer,
        data=str(paths.yolo_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(paths.training_results),
        name=run_name,
        exist_ok=True,
        # Colour is a class attribute, so avoid hue jitter that can swap labels.
        hsv_h=0.0,
        hsv_s=0.20,
        hsv_v=0.20,
        fliplr=0.50,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.20,
        mosaic=0.10,
        erasing=0.10,
        **trainer_overrides,
    )


def validate_detector(paths: ProjectPaths, weights: str | Path, image_size: int = 1024) -> Any:
    """Save detection metrics and confusion matrices under ``Results/validation``."""
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    return model.val(
        data=str(paths.yolo_yaml),
        imgsz=image_size,
        project=str(paths.results / "validation"),
        name="green_yellow_apple",
        exist_ok=True,
    )


def _track_colour(class_id: int) -> tuple[int, int, int]:
    return (40, 180, 40) if class_id == GREEN_APPLE else (0, 220, 240)


def _annotate_tracks(
    frame: np.ndarray, tracks: Iterable[FruitTrack], line_counter: UniqueLineCounter | None
) -> np.ndarray:
    output = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        colour = _track_colour(track.class_id)
        label = f"{CLASS_NAMES[track.class_id]} ID:{track.track_id}"
        if track.depth is not None:
            label += f" d:{track.depth:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    if line_counter is not None:
        start, end = tuple(line_counter.start), tuple(line_counter.end)
        cv2.line(output, start, end, (255, 255, 255), 2)
        cv2.putText(
            output,
            f"Unique apples: {len(line_counter.counted_track_ids)}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )
    return output


def run_depth_aware_tracking(
    paths: ProjectPaths,
    *,
    weights: str | Path,
    video_path: str | Path | None = None,
    adabins: AdaBinsDepth | None = None,
    low_confidence: float = 0.10,
    image_size: int = 1024,
    counting_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    count_direction: str = "positive",
) -> dict[str, Any]:
    """Detect, attach relative depth, track, and save an annotated result video/CSV."""
    input_video = Path(video_path or paths.raw_video)
    info = require_playable_video(input_video)
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(input_video))
    width, height, fps = info["width"], info["height"], info["fps"]
    output_video = paths.inference_results / "tracked_apples.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create result video: {output_video}")
    line_counter = (
        UniqueLineCounter(*counting_line, direction=count_direction)
        if counting_line is not None
        else None
    )
    tracker = DepthAwareByteTracker(low_confidence=low_confidence)
    csv_path = paths.inference_results / "tracks.csv"
    frames_processed = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "frame",
                    "track_id",
                    "class_name",
                    "score",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "relative_depth",
                    "unique_count",
                ),
            )
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = model.predict(frame, conf=low_confidence, imgsz=image_size, verbose=False)[0]
                depth_map = adabins.predict(frame) if adabins is not None else None
                detections = []
                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        class_id = int(box.cls[0].item())
                        if class_id not in (GREEN_APPLE, YELLOW_APPLE):
                            continue
                        bbox = tuple(float(value) for value in box.xyxy[0].tolist())
                        detections.append(
                            FruitDetection(
                                bbox=bbox,
                                score=float(box.conf[0].item()),
                                class_id=class_id,
                                depth=robust_instance_depth(depth_map, bbox),
                            )
                        )
                tracks = tracker.update(detections)
                if line_counter is not None:
                    line_counter.update(tracks)
                unique_count = len(line_counter.counted_track_ids) if line_counter else 0
                for track in tracks:
                    csv_writer.writerow(
                        {
                            "frame": frames_processed,
                            "track_id": track.track_id,
                            "class_name": CLASS_NAMES[track.class_id],
                            "score": round(track.score, 6),
                            "x1": round(float(track.bbox[0]), 2),
                            "y1": round(float(track.bbox[1]), 2),
                            "x2": round(float(track.bbox[2]), 2),
                            "y2": round(float(track.bbox[3]), 2),
                            "relative_depth": "" if track.depth is None else round(track.depth, 6),
                            "unique_count": unique_count,
                        }
                    )
                writer.write(_annotate_tracks(frame, tracks, line_counter))
                frames_processed += 1
    finally:
        capture.release()
        writer.release()
    summary = {
        "input_video": str(input_video),
        "weights": str(weights),
        "frames_processed": frames_processed,
        "relative_depth_enabled": adabins is not None,
        "unique_count": len(line_counter.counted_track_ids) if line_counter else None,
        "output_video": str(output_video),
        "tracks_csv": str(csv_path),
    }
    _write_json(paths.inference_results / "summary.json", summary)
    return summary
