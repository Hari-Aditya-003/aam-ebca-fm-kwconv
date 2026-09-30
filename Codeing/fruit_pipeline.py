"""Reusable helpers for the green-apple detection, depth, tracking, and count workflow.

All generated data is kept below ``Dataset/`` and all reports/artifacts below
``Results/``. The source MP4 is read-only.  The notebook is intentionally the
entry point; this module keeps each notebook cell short and reproducible.
"""

from __future__ import annotations

import csv
import platform
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


CLASS_NAMES = ("green_apple",)
GREEN_APPLE = 0
DEFAULT_VARIANTS = (
    "hflip",
    "vflip",
    "bright",
    "dark",
    "rot90",
    "rot180",
    "rot270",
    "small_rotate",
    "motion_blur",
    "gaussian_noise",
)


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths used by the rebuilt single-green-apple workflow."""

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
    def frames_80fps(self) -> Path:
        return self.dataset / "frames" / "80fps_timeline"

    @property
    def picked_800(self) -> Path:
        return self.dataset / "frames" / "picked_800_uniform"

    @property
    def augmented_800(self) -> Path:
        return self.dataset / "augmentations" / "from_800"

    @property
    def manifests(self) -> Path:
        return self.dataset / "manifests"

    @property
    def yolo(self) -> Path:
        return self.dataset / "yolo_green_apple"

    @property
    def yolo_images(self) -> Path:
        return self.yolo / "images"

    @property
    def yolo_labels(self) -> Path:
        return self.yolo / "labels"

    @property
    def auto_previews(self) -> Path:
        return self.yolo / "auto_label_previews"

    @property
    def data_yaml(self) -> Path:
        return self.yolo / "green_apple.yaml"

    @property
    def hardware_results(self) -> Path:
        return self.results / "hardware"

    @property
    def dataset_results(self) -> Path:
        return self.results / "dataset"

    @property
    def architecture_results(self) -> Path:
        return self.results / "architecture"

    @property
    def training_results(self) -> Path:
        return self.results / "training"

    @property
    def validation_results(self) -> Path:
        return self.results / "validation"

    @property
    def inference_results(self) -> Path:
        return self.results / "inference"

    def ensure_layout(self) -> None:
        folders = (
            self.dataset,
            self.results,
            self.frames_80fps,
            self.picked_800,
            self.augmented_800,
            self.manifests,
            self.auto_previews,
            self.hardware_results,
            self.dataset_results,
            self.architecture_results,
            self.training_results,
            self.validation_results,
            self.inference_results,
        )
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (self.yolo_images / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_labels / split).mkdir(parents=True, exist_ok=True)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root when a notebook is launched from ``Codeing/``."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "Dataset").exists() and (candidate / "Codeing").exists():
            return candidate
    raise FileNotFoundError("Could not find a project root containing Dataset/ and Codeing/.")


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


def _clear_generated_images(folder: Path, pattern: str = "*.jpg") -> None:
    """Delete only generated files in one known pipeline folder."""
    for item in folder.glob(pattern):
        item.unlink()


def video_preflight(video_path: str | Path, *, target_fps: float = 80.0) -> dict[str, Any]:
    """Read input-video metadata and calculate the requested 80-FPS timeline."""
    path = Path(video_path)
    report: dict[str, Any] = {"path": str(path), "exists": path.exists(), "playable": False}
    if not path.exists():
        report["error"] = "Video file does not exist."
        return report
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    readable, _ = capture.read() if opened else (False, None)
    capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    report.update(
        {
            "playable": bool(opened and readable),
            "opened": bool(opened),
            "source_fps": fps,
            "source_frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 6),
            "requested_output_fps": float(target_fps),
            "requested_80fps_timeline_frames": int(round(duration * target_fps)),
            "note": (
                "The source rate is below 80 FPS. An 80-FPS extraction is a resampled timeline; "
                "repeated source-frame indices are recorded in the manifest instead of inventing new images."
                if fps and fps < target_fps
                else "The source rate is at least the requested output rate."
            ),
        }
    )
    if not report["playable"]:
        report["error"] = "OpenCV could not decode a source frame."
    return report


def require_playable_video(video_path: str | Path, *, target_fps: float = 80.0) -> dict[str, Any]:
    report = video_preflight(video_path, target_fps=target_fps)
    if not report["playable"]:
        raise RuntimeError(report.get("error", "The video is not playable."))
    return report


def hardware_report(paths: ProjectPaths) -> dict[str, Any]:
    """Record CUDA/NVIDIA status; the notebook chooses GPU only if it is usable."""
    running_kernel = platform.release()
    # ``nvidiafb`` and laptop backlight helpers do not provide CUDA.  Check the
    # actual proprietary/open compute-driver module specifically.
    nvidia_modules = sorted(Path(f"/lib/modules/{running_kernel}").rglob("nvidia.ko*")) if Path(f"/lib/modules/{running_kernel}").exists() else []
    report: dict[str, Any] = {
        "running_kernel": running_kernel,
        "nvidia_kernel_module_present": bool(nvidia_modules),
    }
    try:
        import torch

        report.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ModuleNotFoundError:
        report["torch_version"] = None
        report["cuda_available"] = False
        report["error"] = "PyTorch is not installed."
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        report["nvidia_smi_returncode"] = completed.returncode
        report["nvidia_smi"] = completed.stdout.strip() or completed.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi"] = str(error)
    report["recommended_device"] = 0 if report.get("cuda_available") else "cpu"
    if not report["cuda_available"]:
        remediation: list[str] = []
        if not report["nvidia_kernel_module_present"]:
            remediation.append(
                "No NVIDIA kernel module is installed for the running kernel. "
                "Install the matching Ubuntu linux-modules-nvidia package, then reboot."
            )
        if str(report.get("torch_version", "")).endswith("+cpu"):
            remediation.append(
                "PyTorch is CPU-only. After nvidia-smi succeeds, install a CUDA-enabled PyTorch build in .venv."
            )
        if not remediation:
            remediation.append("Check nvidia-smi, the NVIDIA driver service, and the CUDA-enabled PyTorch installation.")
        report["cuda_remediation"] = remediation
    _write_json(paths.hardware_results / "runtime.json", report)
    return report


def extract_80fps_timeline(
    paths: ProjectPaths,
    *,
    target_fps: float = 80.0,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Save a timestamp-accurate 80-FPS timeline and source-index manifest.

    A 59.94-FPS input cannot contain 80 unique camera frames each second. This
    function writes one image for each 80-FPS output timestamp and records the
    selected source index, allowing exact accounting of repeated source frames.
    """
    info = require_playable_video(paths.raw_video, target_fps=target_fps)
    source_fps = float(info["source_fps"])
    source_count = int(info["source_frame_count"])
    output_count = int(info["requested_80fps_timeline_frames"])
    if output_count < 1:
        raise ValueError("No 80-FPS timeline frames can be calculated from this video.")
    expected = list(paths.frames_80fps.glob("timeline_*.jpg"))
    manifest_path = paths.manifests / "80fps_timeline.jsonl"
    if not overwrite and len(expected) == output_count and manifest_path.exists():
        return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if overwrite:
        _clear_generated_images(paths.frames_80fps, "timeline_*.jpg")
    capture = cv2.VideoCapture(str(paths.raw_video))
    if not capture.isOpened():
        raise RuntimeError("Could not open the video for 80-FPS extraction.")
    timestamps = np.arange(output_count, dtype=np.float64) / target_fps
    source_indices = np.minimum(np.rint(timestamps * source_fps).astype(np.int64), source_count - 1)
    current_index = -1
    current_frame: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, source_index in enumerate(source_indices):
            desired = int(source_index)
            while current_index < desired:
                ok, frame = capture.read()
                current_index += 1
                if not ok:
                    raise RuntimeError(f"Could not decode source frame {current_index}.")
                current_frame = frame
            if current_frame is None:
                raise RuntimeError("No decoded frame is available for output.")
            output = paths.frames_80fps / f"timeline_{ordinal:05d}_src_{desired:08d}.jpg"
            if not cv2.imwrite(str(output), current_frame):
                raise OSError(f"Could not write {output}")
            rows.append(
                {
                    "image": output.name,
                    "timeline_index": ordinal,
                    "timestamp_seconds": round(float(timestamps[ordinal]), 6),
                    "source_frame_index": desired,
                    "source_frame_repeated": bool(ordinal > 0 and desired == int(source_indices[ordinal - 1])),
                }
            )
    finally:
        capture.release()
    _write_jsonl(manifest_path, rows)
    _write_json(
        paths.dataset_results / "80fps_extraction_summary.json",
        {
            **info,
            "written_frames": len(rows),
            "unique_source_frames": len(set(int(value) for value in source_indices)),
            "repeated_source_timestamps": int(sum(rows[i]["source_frame_repeated"] for i in range(len(rows)))),
            "output_folder": str(paths.frames_80fps),
        },
    )
    return rows


def pick_uniform_800(paths: ProjectPaths, *, count: int = 800, overwrite: bool = False) -> list[dict[str, Any]]:
    """Copy exactly ``count`` uniformly distributed 80-FPS timeline images."""
    timeline = sorted(paths.frames_80fps.glob("timeline_*.jpg"))
    if len(timeline) < count:
        raise ValueError(f"Need at least {count} timeline frames; found {len(timeline)}.")
    existing = list(paths.picked_800.glob("pick_*.jpg"))
    manifest_path = paths.manifests / "picked_800_uniform.jsonl"
    if not overwrite and len(existing) == count and manifest_path.exists():
        return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if overwrite:
        _clear_generated_images(paths.picked_800, "pick_*.jpg")
    indices = np.unique(np.linspace(0, len(timeline) - 1, count, dtype=np.int64))
    if len(indices) != count:
        raise RuntimeError("Uniform selection did not produce the requested number of unique positions.")
    rows: list[dict[str, Any]] = []
    for ordinal, timeline_index in enumerate(indices):
        source = timeline[int(timeline_index)]
        target = paths.picked_800 / f"pick_{ordinal:04d}_{source.name}"
        shutil.copy2(source, target)
        rows.append(
            {
                "image": target.name,
                "pick_index": ordinal,
                "timeline_image": source.name,
                "timeline_index": int(timeline_index),
            }
        )
    _write_jsonl(manifest_path, rows)
    return rows


def _brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _small_rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _motion_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float = 10.0) -> np.ndarray:
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_picked_800(
    paths: ProjectPaths,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, int]:
    """Create ten documented visual augmentations for every selected frame.

    Labels are intentionally *not* transformed here: the requested workflow
    automatically labels each final split image after the split stage.
    """
    originals = sorted(paths.picked_800.glob("pick_*.jpg"))
    if len(originals) != 800:
        raise ValueError(f"Expected exactly 800 picked frames; found {len(originals)}.")
    expected = len(originals) * len(DEFAULT_VARIANTS)
    existing = list(paths.augmented_800.glob("*__*.jpg"))
    manifest_path = paths.manifests / "augmentation_manifest.jsonl"
    if not overwrite and len(existing) == expected and manifest_path.exists():
        return {"originals": len(originals), "variants_per_original": len(DEFAULT_VARIANTS), "augmented": len(existing), "total": len(originals) + len(existing)}
    if overwrite:
        _clear_generated_images(paths.augmented_800, "*__*.jpg")
    random_state = random.Random(seed)
    noise_state = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for original in originals:
        image = cv2.imread(str(original))
        if image is None:
            raise OSError(f"Could not read {original}")
        random_angle = random_state.uniform(-12.0, 12.0)
        variants = {
            "hflip": cv2.flip(image, 1),
            "vflip": cv2.flip(image, 0),
            "bright": _brightness(image, 1.20),
            "dark": _brightness(image, 0.80),
            "rot90": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            "rot180": cv2.rotate(image, cv2.ROTATE_180),
            "rot270": cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            "small_rotate": _small_rotate(image, random_angle),
            "motion_blur": _motion_blur(image),
            "gaussian_noise": _gaussian_noise(image, noise_state),
        }
        for name, transformed in variants.items():
            output = paths.augmented_800 / f"{original.stem}__{name}.jpg"
            if not cv2.imwrite(str(output), transformed):
                raise OSError(f"Could not write {output}")
            rows.append({"image": output.name, "source_image": original.name, "variant": name, "random_angle": round(random_angle, 4) if name == "small_rotate" else None})
    _write_jsonl(manifest_path, rows)
    summary = {"originals": len(originals), "variants_per_original": len(DEFAULT_VARIANTS), "augmented": len(rows), "total": len(originals) + len(rows)}
    _write_json(paths.dataset_results / "augmentation_summary.json", summary)
    return summary


def _split_counts(total: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("At least three original frame groups are required.")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Invalid split fractions.")
    train = int(total * train_fraction)
    val = int(total * val_fraction)
    return train, val, total - train - val


def _write_data_yaml(paths: ProjectPaths) -> None:
    root = json.dumps(str(paths.yolo.resolve()))
    paths.data_yaml.write_text(
        "\n".join((f"path: {root}", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: green_apple", "")),
        encoding="utf-8",
    )


def build_grouped_dataset_split(
    paths: ProjectPaths,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    overwrite: bool = False,
) -> dict[str, int]:
    """Split 800 original-frame groups and their ten variants into 70/10/20.

    Each original and all ten derivatives remain in the same split. This gives
    exact 70/10/20 frame-group proportions while avoiding augmentation leakage.
    Automatic labels are deliberately created only after this function.
    """
    originals = sorted(paths.picked_800.glob("pick_*.jpg"))
    if len(originals) != 800:
        raise ValueError(f"Expected 800 originals; found {len(originals)}.")
    groups: list[tuple[Path, list[Path]]] = []
    for original in originals:
        derivatives = sorted(paths.augmented_800.glob(f"{original.stem}__*.jpg"))
        if len(derivatives) != len(DEFAULT_VARIANTS):
            raise ValueError(f"{original.name} has {len(derivatives)} variants, not {len(DEFAULT_VARIANTS)}.")
        groups.append((original, [original, *derivatives]))
    train_groups, val_groups, test_groups = _split_counts(len(groups), train_fraction, val_fraction)
    assignments = {
        "train": groups[:train_groups],
        "val": groups[train_groups : train_groups + val_groups],
        "test": groups[train_groups + val_groups :],
    }
    if overwrite:
        for split in assignments:
            _clear_generated_images(paths.yolo_images / split, "*.jpg")
            _clear_generated_images(paths.yolo_labels / split, "*.txt")
            _clear_generated_images(paths.auto_previews / split, "*.jpg")
    rows: list[dict[str, Any]] = []
    for split, frame_groups in assignments.items():
        image_folder = paths.yolo_images / split
        for original, members in frame_groups:
            for member in members:
                target = image_folder / member.name
                if not target.exists() or overwrite:
                    shutil.copy2(member, target)
                rows.append({"image": member.name, "source_image": original.name, "split": split, "variant": "original" if member == original else member.stem.rsplit("__", 1)[-1]})
    _write_jsonl(paths.manifests / "grouped_split.jsonl", rows)
    _write_data_yaml(paths)
    image_counts = {split: sum(len(members) for _, members in frame_groups) for split, frame_groups in assignments.items()}
    result = {"source_groups": len(groups), "train_groups": train_groups, "val_groups": val_groups, "test_groups": test_groups, **{f"{split}_images": count for split, count in image_counts.items()}}
    _write_json(paths.dataset_results / "split_summary.json", result)
    return result


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    score: float | None = None

    def xyxy(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        width, height = self.width * image_width, self.height * image_height
        center_x, center_y = self.x_center * image_width, self.y_center * image_height
        return center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2


def xyxy_to_yolo(class_id: int, box: Sequence[float], image_width: int, image_height: int, score: float | None = None) -> YoloBox:
    left, top, right, bottom = box
    left, right = np.clip((left, right), 0, image_width)
    top, bottom = np.clip((top, bottom), 0, image_height)
    width, height = right - left, bottom - top
    if width <= 1 or height <= 1:
        raise ValueError("Bounding box is empty after clipping.")
    return YoloBox(class_id, (left + right) / (2 * image_width), (top + bottom) / (2 * image_height), width / image_width, height / image_height, score)


def read_yolo_labels(path: str | Path) -> list[YoloBox]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.read_text(encoding="utf-8").strip():
        return []
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{file_path}:{line_number} is not a YOLO label row.")
        class_id, x_center, y_center, width, height = int(values[0]), *(float(value) for value in values[1:])
        if class_id != GREEN_APPLE or not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{file_path}:{line_number} contains an invalid green-apple label.")
        boxes.append(YoloBox(class_id, x_center, y_center, width, height))
    return boxes


def write_yolo_labels(path: str | Path, boxes: Iterable[YoloBox]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}" for box in boxes), encoding="utf-8")


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _nms(candidates: list[tuple[tuple[float, float, float, float], float]], threshold: float = 0.50) -> list[tuple[tuple[float, float, float, float], float]]:
    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for box, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(_iou(box, previous_box) < threshold for previous_box, _ in kept):
            kept.append((box, score))
    return kept


def draw_yolo_boxes(image: np.ndarray, boxes: Iterable[YoloBox]) -> np.ndarray:
    """Draw all single-class auto labels for dataset diagnostics."""
    output = image.copy()
    for box in boxes:
        left, top, right, bottom = (int(round(value)) for value in box.xyxy(image.shape[1], image.shape[0]))
        cv2.rectangle(output, (left, top), (right, bottom), (40, 210, 40), 2)
        cv2.putText(output, "green_apple", (left, max(20, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 210, 40), 2, cv2.LINE_AA)
    return output


def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("Install Ultralytics before automatic labelling/training: pip install ultralytics") from error
    return YOLO


def _model_apple_ids(model: Any) -> set[int]:
    names = {int(index): str(name).lower().strip().replace(" ", "_") for index, name in model.names.items()}
    apple_ids = {index for index, name in names.items() if name in {"apple", "green_apple"}}
    if not apple_ids:
        raise RuntimeError(f"The selected auto-label model has no apple class: {names}")
    return apple_ids


def _model_candidates(model: Any, image: np.ndarray, *, apple_ids: set[int], confidence: float, image_size: int, device: str | int) -> list[tuple[tuple[float, float, float, float], float]]:
    result = model.predict(image, conf=confidence, imgsz=image_size, device=device, verbose=False)[0]
    candidates: list[tuple[tuple[float, float, float, float], float]] = []
    if result.boxes is not None:
        for detected in result.boxes:
            if int(detected.cls[0].item()) not in apple_ids:
                continue
            candidates.append((tuple(float(value) for value in detected.xyxy[0].tolist()), float(detected.conf[0].item())))
    return candidates


def auto_label_after_split(
    paths: ProjectPaths,
    *,
    model_name: str | Path = "yolov8x.pt",
    confidence: float = 0.15,
    image_size: int = 1280,
    device: str | int = "cpu",
    tiled: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Automatically label every final split image as class ``green_apple``.

    The COCO apple detector proposes every apple it sees. Tiled 2×2 inference
    complements full-frame inference for small/overlapping orchard fruit, then
    NMS merges duplicate boxes. No manual label file is read or required.
    """
    YOLO = _require_ultralytics()
    model = YOLO(str(model_name))
    apple_ids = _model_apple_ids(model)
    all_rows: list[dict[str, Any]] = []
    totals = {"images": 0, "boxes": 0, "empty_images": 0, "train_images": 0, "val_images": 0, "test_images": 0}
    for split in ("train", "val", "test"):
        images = sorted((paths.yolo_images / split).glob("*.jpg"))
        if not images:
            raise ValueError(f"No split images found in {paths.yolo_images / split}.")
        if overwrite:
            _clear_generated_images(paths.yolo_labels / split, "*.txt")
            _clear_generated_images(paths.auto_previews / split, "*.jpg")
        for image_path in images:
            label_path = paths.yolo_labels / split / f"{image_path.stem}.txt"
            preview_path = paths.auto_previews / split / image_path.name
            if label_path.exists() and preview_path.exists() and not overwrite:
                boxes = read_yolo_labels(label_path)
                all_rows.append({"image": image_path.name, "split": split, "status": "reused", "boxes": len(boxes)})
                totals["images"] += 1
                totals[f"{split}_images"] += 1
                totals["boxes"] += len(boxes)
                totals["empty_images"] += int(not boxes)
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise OSError(f"Could not read {image_path}")
            candidates = _model_candidates(model, image, apple_ids=apple_ids, confidence=confidence, image_size=image_size, device=device)
            if tiled:
                height, width = image.shape[:2]
                overlap_x, overlap_y = int(width * 0.12), int(height * 0.12)
                x_ranges = ((0, width // 2 + overlap_x), (width // 2 - overlap_x, width))
                y_ranges = ((0, height // 2 + overlap_y), (height // 2 - overlap_y, height))
                for top, bottom in y_ranges:
                    for left, right in x_ranges:
                        tile = image[top:bottom, left:right]
                        for (x1, y1, x2, y2), score in _model_candidates(model, tile, apple_ids=apple_ids, confidence=confidence, image_size=image_size, device=device):
                            candidates.append(((x1 + left, y1 + top, x2 + left, y2 + top), score))
            selected = _nms(candidates)
            boxes: list[YoloBox] = []
            for candidate, score in selected:
                try:
                    boxes.append(xyxy_to_yolo(GREEN_APPLE, candidate, image.shape[1], image.shape[0], score))
                except ValueError:
                    continue
            write_yolo_labels(label_path, boxes)
            cv2.imwrite(str(preview_path), draw_yolo_boxes(image, boxes))
            all_rows.append({"image": image_path.name, "split": split, "status": "generated", "boxes": len(boxes), "mean_detector_confidence": round(float(np.mean([box.score for box in boxes if box.score is not None]), 6) if boxes else 0.0, 6)})
            totals["images"] += 1
            totals[f"{split}_images"] += 1
            totals["boxes"] += len(boxes)
            totals["empty_images"] += int(not boxes)
    _write_jsonl(paths.manifests / "automatic_green_apple_labels.jsonl", all_rows)
    report = {"model": str(model_name), "confidence": confidence, "image_size": image_size, "tiled_inference": tiled, "class": CLASS_NAMES[0], **totals}
    _write_json(paths.dataset_results / "automatic_label_summary.json", report)
    return report


def validate_dataset(paths: ProjectPaths) -> dict[str, Any]:
    """Validate image/label parity and class counts after automatic labelling."""
    report: dict[str, Any] = {"splits": {}, "class_counts": {"green_apple": 0}, "valid": False}
    for split in ("train", "val", "test"):
        images = sorted((paths.yolo_images / split).glob("*.jpg"))
        missing_labels: list[str] = []
        boxes = 0
        empty_labels = 0
        for image_path in images:
            label_path = paths.yolo_labels / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue
            labels = read_yolo_labels(label_path)
            boxes += len(labels)
            empty_labels += int(not labels)
            report["class_counts"]["green_apple"] += len(labels)
        report["splits"][split] = {"images": len(images), "labels": len(images) - len(missing_labels), "boxes": boxes, "empty_labels": empty_labels, "missing_labels": missing_labels}
    report["valid"] = all(
        part["images"] > 0 and not part["missing_labels"]
        for part in report["splits"].values()
    )
    _write_json(paths.dataset_results / "dataset_audit.json", report)
    return report


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ModuleNotFoundError as error:
        raise RuntimeError("Install PyTorch before constructing the custom feature modules.") from error
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
            self.colour = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1))
            self.shape = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            self.scale = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), nn.Conv2d(channels, channels, 1))
            self.weights = nn.Parameter(torch.zeros(3))

        def forward(self, features: Any) -> Any:
            colour = self.colour(features).expand_as(features)
            shape = self.shape(features)
            scale = self.scale(features)
            alpha = torch.softmax(self.weights, dim=0)
            return features * torch.sigmoid(alpha[0] * colour + alpha[1] * shape + alpha[2] * scale)


    class EBCA(nn.Module):
        """Efficient bidirectional cross-attention between P3 and P4."""

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
            output_p3 = self.p3_norm(p3 + p3_gate * p4_up)
            p3_down = F.adaptive_avg_pool2d(self.p3_to_p4(p3), p4.shape[-2:])
            p4_gate = torch.sigmoid(self.p4_query(p4) * p3_down / math.sqrt(p4.shape[1]))
            output_p4 = self.p4_norm(p4 + p4_gate * p3_down)
            return output_p3, output_p4


    class FM(nn.Module):
        """Focal modulation with local/global context for partially occluded apples."""

        def __init__(self, channels: int, levels: int = 3) -> None:
            super().__init__()
            self.query = nn.Conv2d(channels, channels, 1)
            self.context = nn.ModuleList([nn.Conv2d(channels, channels, 3 + 2 * level, padding=1 + level, groups=channels) for level in range(levels)])
            self.gates = nn.Conv2d(channels, levels + 1, 1)
            self.modulator = nn.Conv2d(channels, channels, 1)
            self.output = nn.Conv2d(channels, channels, 1)

        def forward(self, features: Any) -> Any:
            query = self.query(features)
            gates = torch.softmax(self.gates(features), dim=1)
            context: Any = 0
            current = features
            for level, layer in enumerate(self.context):
                current = layer(current)
                context = context + gates[:, level : level + 1] * current
            context = context + gates[:, -1:] * F.adaptive_avg_pool2d(features, 1)
            return self.output(query * self.modulator(context))


    class KWConv(nn.Module):
        """Kernel warehouse mixing 3×3, 5×5, and 7×7 kernels per input."""

        def __init__(self, channels: int, kernels: Sequence[int] = (3, 5, 7)) -> None:
            super().__init__()
            self.branches = nn.ModuleList([nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False) for kernel in kernels])
            self.selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, len(kernels), 1))
            self.norm = nn.BatchNorm2d(channels)
            self.activation = nn.SiLU()

        def forward(self, features: Any) -> Any:
            weights = torch.softmax(self.selector(features), dim=1)
            mixed = sum(weights[:, index : index + 1] * branch(features) for index, branch in enumerate(self.branches))
            return self.activation(self.norm(mixed))


    class ProposedEnhancementNeck(nn.Module):
        """AAM → EBCA → FM → KWConv on YOLO P3/P4/P5 features."""

        def __init__(self, p3_channels: int, p4_channels: int, p5_channels: int) -> None:
            super().__init__()
            self.aam = AAM(p4_channels)
            self.ebca = EBCA(p3_channels, p4_channels)
            self.fm = FM(p4_channels)
            self.kw_p3, self.kw_p4, self.kw_p5 = KWConv(p3_channels), KWConv(p4_channels), KWConv(p5_channels)

        def forward(self, p3: Any, p4: Any, p5: Any) -> tuple[Any, Any, Any]:
            p4 = self.aam(p4)
            p3, p4 = self.ebca(p3, p4)
            p4 = self.fm(p4)
            return self.kw_p3(p3), self.kw_p4(p4), self.kw_p5(p5)


    class EnhancedDetectHead(nn.Module):
        """Wrap Ultralytics Detect so it consumes the proposed enhanced P3/P4/P5."""

        def __init__(self, detect: Any, channels: Sequence[int]) -> None:
            super().__init__()
            self.detect = detect
            self.enhancement = ProposedEnhancementNeck(*channels)
            self.i, self.f = getattr(detect, "i", None), getattr(detect, "f", None)
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
    """Verify P3/P4/P5 tensor interfaces and save an architecture report."""
    if torch is None:
        _require_torch()
    model = ProposedEnhancementNeck(64, 128, 256).to(device).eval()
    with torch.inference_mode():
        outputs = model(torch.zeros(1, 64, 80, 80, device=device), torch.zeros(1, 128, 40, 40, device=device), torch.zeros(1, 256, 20, 20, device=device))
    report = {"device": device, "output_shapes": [list(output.shape) for output in outputs], "module_order": ["AAM", "EBCA", "FM", "KWConv"]}
    _write_json(paths.architecture_results / "module_smoke_test.json", report)
    return report


def patch_yolov8_detection_head(yolo: Any) -> Any:
    """Insert the proposed feature neck immediately before a standard YOLOv8 Detect head."""
    if torch is None:
        _require_torch()
    candidate = getattr(yolo, "model", None)
    detection_model = candidate if candidate is not None and hasattr(candidate, "model") else yolo
    modules = getattr(detection_model, "model", None)
    if modules is None or not len(modules):
        raise TypeError("Expected an Ultralytics DetectionModel or YOLO object.")
    head = modules[-1]
    if isinstance(head, EnhancedDetectHead):
        return yolo
    try:
        channels = tuple(int(branch[0].conv.in_channels) for branch in head.cv2)
    except (AttributeError, TypeError, IndexError) as error:
        raise TypeError("Expected a standard three-scale Ultralytics Detect head.") from error
    if len(channels) != 3:
        raise ValueError(f"Expected P3/P4/P5 channels; got {channels}.")
    modules[-1] = EnhancedDetectHead(head, channels)
    return yolo


def train_enhanced_yolov8(
    paths: ProjectPaths,
    *,
    model_name: str = "yolov8x.pt",
    epochs: int = 100,
    image_size: int = 1024,
    batch: int | float = -1,
    device: str | int = 0,
    workers: int = 4,
    run_name: str = "aam_ebca_fm_kwconv_green",
    **overrides: Any,
) -> Any:
    """Train the full AAM–EBCA–FM–KWConv enhanced YOLOv8 model.

    The custom trainer patches the freshly reconstructed one-class model. It
    therefore saves the proposed neck inside the final checkpoint, not only in
    a notebook-side test object.
    """
    audit = validate_dataset(paths)
    if not audit["valid"]:
        raise ValueError("Run automatic labelling and resolve dataset audit errors before training.")
    YOLO = _require_ultralytics()
    from ultralytics.models.yolo.detect import DetectionTrainer

    class GreenAppleEnhancedTrainer(DetectionTrainer):
        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            patch_yolov8_detection_head(model)
            return model

    model = YOLO(model_name)
    return model.train(
        trainer=GreenAppleEnhancedTrainer,
        data=str(paths.data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=workers,
        project=str(paths.training_results),
        name=run_name,
        exist_ok=True,
        patience=0,  # complete all requested epochs; no early-stop interruption
        hsv_h=0.0,  # colour represents the single green-apple class
        hsv_s=0.15,
        hsv_v=0.20,
        fliplr=0.50,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.20,
        mosaic=0.10,
        erasing=0.10,
        **overrides,
    )


def validate_model(paths: ProjectPaths, weights: str | Path, *, image_size: int = 1024, device: str | int = 0) -> Any:
    """Save detection validation outputs (precision, recall, mAP and curves)."""
    YOLO = _require_ultralytics()
    return YOLO(str(weights)).val(data=str(paths.data_yaml), imgsz=image_size, device=device, project=str(paths.validation_results), name="green_apple", exist_ok=True)


def summarize_metrics(paths: ProjectPaths, run_directory: str | Path) -> dict[str, Any]:
    """Extract last-epoch precision, recall, mAP and calculated F1 into Results."""
    run_path = Path(run_directory)
    csv_path = run_path / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training results CSV was not found: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training results CSV is empty.")
    row = {key.strip(): value for key, value in rows[-1].items()}
    def number(key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None
    precision = number("metrics/precision(B)")
    recall = number("metrics/recall(B)")
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
    report = {"run_directory": str(run_path), "epoch": int(float(row.get("epoch", len(rows) - 1))), "precision": precision, "recall": recall, "f1": f1, "map50": number("metrics/mAP50(B)"), "map50_95": number("metrics/mAP50-95(B)"), "result_csv": str(csv_path)}
    _write_json(paths.validation_results / "precision_recall_f1_summary.json", report)
    return report


class AdaBinsDepth:
    """Official AdaBins adapter; output is relative depth without orchard calibration."""

    def __init__(self, repository: str | Path, dataset: str = "kitti", device: str | None = None) -> None:
        repository_path = Path(repository).expanduser().resolve()
        if not (repository_path / "infer.py").exists():
            raise FileNotFoundError(f"AdaBins repository not found: {repository_path}")
        if str(repository_path) not in sys.path:
            sys.path.insert(0, str(repository_path))
        try:
            from infer import InferenceHelper
        except ModuleNotFoundError as error:
            raise RuntimeError("AdaBins dependencies are unavailable in this notebook kernel.") from error
        if torch is None:
            _require_torch()
        selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        previous = Path.cwd()
        try:
            os.chdir(repository_path)
            self.helper = InferenceHelper(dataset=dataset, device=selected_device)
        finally:
            os.chdir(previous)

    def predict(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        _, depth = self.helper.predict(tensor.to(self.helper.device))
        depth_map = np.asarray(depth).squeeze().astype(np.float32)
        return cv2.resize(depth_map, (bgr_frame.shape[1], bgr_frame.shape[0]), interpolation=cv2.INTER_LINEAR)


def robust_instance_depth(depth_map: np.ndarray | None, bbox: Sequence[float], trim_fraction: float = 0.10) -> float | None:
    """Return a robust central depth statistic for one detection box."""
    if depth_map is None:
        return None
    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    margin_x, margin_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    left, right = max(0, int(x1 + margin_x)), min(width, int(x2 - margin_x))
    top, bottom = max(0, int(y1 + margin_y)), min(height, int(y2 - margin_y))
    values = depth_map[top:bottom, left:right].reshape(-1) if right > left and bottom > top else np.array([])
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return None
    values.sort()
    trim = int(len(values) * trim_fraction)
    if trim and len(values) > 2 * trim:
        values = values[trim:-trim]
    return float(np.median(values))


@dataclass(frozen=True)
class FruitDetection:
    bbox: tuple[float, float, float, float]
    score: float
    depth: float | None = None


@dataclass
class FruitTrack:
    track_id: int
    bbox: np.ndarray
    score: float
    depth: float | None
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    updated: bool = True

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
    """ByteTrack-style high/low confidence association with relative-depth gating."""

    def __init__(self, *, high_confidence: float = 0.50, low_confidence: float = 0.10, iou_threshold: float = 0.10, max_cost: float = 0.70, depth_gate: float = 0.40, track_buffer: int = 30, min_hits: int = 3) -> None:
        self.high_confidence, self.low_confidence = high_confidence, low_confidence
        self.iou_threshold, self.max_cost, self.depth_gate = iou_threshold, max_cost, depth_gate
        self.track_buffer, self.min_hits = track_buffer, min_hits
        self.tracks: list[FruitTrack] = []
        self.next_id = 1

    def _cost(self, track: FruitTrack, detection: FruitDetection) -> float:
        overlap = _iou(track.bbox, detection.bbox)
        if overlap < self.iou_threshold:
            return float("inf")
        depth_cost = 0.0
        if track.depth is not None and detection.depth is not None:
            difference = abs(track.depth - detection.depth) / max(abs(track.depth), abs(detection.depth), 1e-6)
            if difference > self.depth_gate:
                return float("inf")
            depth_cost = difference / self.depth_gate
        scale = max(1.0, math.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]))
        detection_center = np.array([(detection.bbox[0] + detection.bbox[2]) / 2, (detection.bbox[1] + detection.bbox[3]) / 2])
        motion_cost = min(1.0, float(np.linalg.norm(track.center - detection_center) / (3 * scale)))
        return 0.65 * (1 - overlap) + 0.20 * depth_cost + 0.15 * motion_cost

    def _associate(self, track_indices: list[int], detections: list[FruitDetection], max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not detections:
            return [], track_indices.copy(), list(range(len(detections)))
        costs = np.full((len(track_indices), len(detections)), 1e6, dtype=np.float64)
        for row, track_index in enumerate(track_indices):
            for column, detection in enumerate(detections):
                costs[row, column] = self._cost(self.tracks[track_index], detection)
        try:
            from scipy.optimize import linear_sum_assignment
            rows, columns = linear_sum_assignment(costs)
        except ModuleNotFoundError:
            rows, columns = np.array([], dtype=int), np.array([], dtype=int)
        matches, used_tracks, used_detections = [], set(), set()
        for row, column in zip(rows, columns):
            if costs[row, column] <= max_cost:
                matches.append((track_indices[int(row)], int(column)))
                used_tracks.add(track_indices[int(row)])
                used_detections.add(int(column))
        return matches, [item for item in track_indices if item not in used_tracks], [item for item in range(len(detections)) if item not in used_detections]

    def update(self, detections: Iterable[FruitDetection]) -> list[FruitTrack]:
        active = [item for item in detections if item.score >= self.low_confidence]
        for track in self.tracks:
            track.predict()
        high = [item for item in active if item.score >= self.high_confidence]
        low = [item for item in active if item.score < self.high_confidence]
        matches, unmatched_tracks, unmatched_high = self._associate(list(range(len(self.tracks))), high, self.max_cost)
        for track_index, detection_index in matches:
            self.tracks[track_index].update(high[detection_index])
        low_matches, _, _ = self._associate(unmatched_tracks, low, min(0.85, self.max_cost + 0.10))
        for track_index, detection_index in low_matches:
            self.tracks[track_index].update(low[detection_index])
        for detection_index in unmatched_high:
            detection = high[detection_index]
            self.tracks.append(FruitTrack(self.next_id, np.asarray(detection.bbox, dtype=np.float32), detection.score, detection.depth))
            self.next_id += 1
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.track_buffer]
        return [track for track in self.tracks if track.updated and track.hits >= self.min_hits]


class UniqueLineCounter:
    """Count each confirmed track once when crossing a virtual line."""

    def __init__(self, start: tuple[int, int], end: tuple[int, int], direction: str = "both") -> None:
        if direction not in {"positive", "negative", "both"}:
            raise ValueError("direction must be positive, negative, or both")
        self.start, self.end, self.direction = np.array(start), np.array(end), direction
        self.counted_ids: set[int] = set()
        self.previous_sides: dict[int, float] = {}

    def update(self, tracks: Iterable[FruitTrack]) -> list[int]:
        newly_counted = []
        for track in tracks:
            point = track.center
            side = float(np.cross(self.end - self.start, point - self.start))
            previous = self.previous_sides.get(track.track_id)
            self.previous_sides[track.track_id] = side
            if previous is None or track.track_id in self.counted_ids or previous * side >= 0:
                continue
            valid = self.direction == "both" or (self.direction == "positive" and side > 0) or (self.direction == "negative" and side < 0)
            if valid:
                self.counted_ids.add(track.track_id)
                newly_counted.append(track.track_id)
        return newly_counted


def run_depth_aware_tracking(
    paths: ProjectPaths,
    *,
    weights: str | Path,
    adabins: AdaBinsDepth | None = None,
    confidence: float = 0.10,
    image_size: int = 1024,
    device: str | int = 0,
    line: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Run enhanced YOLO, optional AdaBins depth, ByteTrack, and unique counting."""
    info = require_playable_video(paths.raw_video)
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(paths.raw_video))
    output = paths.inference_results / "tracked_green_apples.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(info["source_fps"]), (int(info["width"]), int(info["height"])))
    if not writer.isOpened():
        raise OSError(f"Could not create {output}")
    tracker = DepthAwareByteTracker(low_confidence=confidence)
    counter = UniqueLineCounter(*line, direction="both") if line else None
    csv_path = paths.inference_results / "tracks.csv"
    frame_index = 0
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fields = ("frame", "track_id", "score", "x1", "y1", "x2", "y2", "relative_depth", "unique_count")
            csv_writer = csv.DictWriter(handle, fieldnames=fields)
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = model.predict(frame, conf=confidence, imgsz=image_size, device=device, verbose=False)[0]
                depth_map = adabins.predict(frame) if adabins is not None else None
                detections: list[FruitDetection] = []
                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        if int(box.cls[0].item()) != GREEN_APPLE:
                            continue
                        bbox = tuple(float(value) for value in box.xyxy[0].tolist())
                        detections.append(FruitDetection(bbox, float(box.conf[0].item()), robust_instance_depth(depth_map, bbox)))
                tracks = tracker.update(detections)
                if counter:
                    counter.update(tracks)
                annotated = frame.copy()
                for track in tracks:
                    x1, y1, x2, y2 = (int(value) for value in track.bbox)
                    text = f"green_apple ID:{track.track_id}" + (f" d:{track.depth:.2f}" if track.depth is not None else "")
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 210, 40), 2)
                    cv2.putText(annotated, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 210, 40), 2)
                    csv_writer.writerow({"frame": frame_index, "track_id": track.track_id, "score": round(track.score, 6), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "relative_depth": "" if track.depth is None else round(track.depth, 6), "unique_count": len(counter.counted_ids) if counter else ""})
                if counter:
                    cv2.line(annotated, tuple(counter.start), tuple(counter.end), (255, 255, 255), 2)
                    cv2.putText(annotated, f"Unique apples: {len(counter.counted_ids)}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                writer.write(annotated)
                frame_index += 1
    finally:
        capture.release()
        writer.release()
    summary = {"input_video": str(paths.raw_video), "frames_processed": frame_index, "weights": str(weights), "relative_depth_enabled": adabins is not None, "unique_count": len(counter.counted_ids) if counter else None, "output_video": str(output), "tracks_csv": str(csv_path)}
    _write_json(paths.inference_results / "tracking_summary.json", summary)
    return summary
