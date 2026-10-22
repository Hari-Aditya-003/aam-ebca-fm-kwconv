#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# Executable source export of ``AAM, EBCA, FM, KWConv 100.ipynb``.
# Expensive stages remain guarded by the same RUN_* switches as the notebook.

# # Enhanced Green-Apple Detection, Relative Depth, and Duplicate-Safe Tracking
#
# This notebook follows the requested workflow: **80-FPS timeline → 800 uniform frames → augmentation → grouped 70/10/20 split → automatic green-apple labels → enhanced YOLOv8 (AAM, EBCA, FM, KWConv) → AdaBins relative depth → depth-aware ByteTrack counting**.
#
# The input is 59.94 FPS. The 80-FPS stage resamples timestamps and records repeated source-frame indices rather than fabricating new camera content. Generated data is local under `Dataset/` and `Results/`; it is excluded from Git.

# ## How to run — strict step-by-step order
#
# Run the notebook from top to bottom. Do **not** enable later stages until the previous stage has completed and its stated output path exists.
#
# | Step | Run this notebook section | Required input | Main output path |
# |---:|---|---|---|
# | 01 | Environment, GPU, paths | `.venv`, NVIDIA driver | `Results/hardware/runtime.json` |
# | 02 | Video preflight | `Dataset/DJI_0418.MP4` | metadata displayed only |
# | 03 | 80-FPS extraction | playable source video | `Dataset/frames/80fps_timeline/` |
# | 04 | Uniform selection | 5,105 timeline images | `Dataset/frames/picked_800_uniform/` |
# | 05 | Augmentation | 800 picked images | `Dataset/augmentations/from_800/` |
# | 06 | Inventory | frame folders | displayed 5,105 / 800 / 8,000 / 8,800 counts |
# | 07 | Grouped split | originals + variants | `Dataset/yolo_green_apple/images/{train,val,test}/` |
# | 08 | Automatic labels | split images | `Dataset/yolo_green_apple/labels/` and previews |
# | 09 | Label audit | split images and labels | `Results/dataset/dataset_audit.json` |
# | 10 | AAM, EBCA, FM, KWConv checks | CUDA/PyTorch | `Results/architecture/module_smoke_test.json` |
# | 11 | Enhanced YOLOv8 training | audited YAML | `Results/training/.../weights/best.pt` |
# | 12 | Final metrics | `best.pt` | `Results/validation/precision_recall_f1_summary.json` |
# | 13 | AdaBins + ByteTrack | `best.pt`, optional AdaBins | `Results/inference/` |
#
# For a stage that writes data, set **only that cell’s** `RUN_...` variable to `True`, run it, confirm the output count/path, then set it back to `False`. This keeps the procedure reproducible and prevents accidental re-creation of large image sets or model runs.

# ## Methodology mapping: notebook steps to the proposed architecture
#
# **Data path:** RGB video → 80-FPS timestamp timeline → uniform 800 frames → ten visual augmentations per frame → grouped 70/10/20 image split → automatic single-class green-apple YOLO labels.
# **Detection path:** YOLOv8 P3/P4/P5 features → **AAM** (colour/shape/scale attributes) → **EBCA** (P3↔P4 cross-scale exchange) → **FM** (contextual occlusion modulation) → **KWConv** (adaptive multi-scale kernels) → three-scale Detect head.
# **Depth/count path:** RGB frame → AdaBins relative depth → per-box robust depth → high/low confidence association + motion + IoU + depth-aware ByteTrack → persistent fruit IDs → one-time line-crossing count.
# The implementation cells `02A`–`02G` contain this code in the notebook; workflow cells `03`–`18` execute it in the order listed above.

# In[1]:


# %% 01) Imports and project paths
from pathlib import Path
import json

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'Codeing').is_dir():
    PROJECT_ROOT = Path('/home/hari/Desktop/AAM, EBCA, FM, KWConv').resolve()

print('Project root:', PROJECT_ROOT)
print('Input video :', PROJECT_ROOT / 'Dataset' / 'DJI_0418.MP4')
print('Dataset dir :', PROJECT_ROOT / 'Dataset')
print('Results dir :', PROJECT_ROOT / 'Results')


# ## Notebook-only implementation — core paths
#
# **Purpose:** define the project’s one-class green-apple workflow without importing any local `.py` file.
# **Reads:** the notebook location and `Dataset/DJI_0418.MP4`.
# **Creates when called later:** `Dataset/` and `Results/` subfolders only; it never modifies the source video.
# **Code coverage:** project paths, JSON manifests, runtime report, and video preflight helpers.

# In[2]:


# %% 02A) Core paths, manifests, and video metadata
"""Reusable helpers for the green-apple detection, depth, tracking, and count workflow.

All generated data is kept below ``Dataset/`` and all reports/artifacts below
``Results/``. The source MP4 is read-only.  The notebook is intentionally the
entry point; this module keeps each notebook cell short and reproducible.
"""

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


# ## Notebook-only implementation — frame extraction and uniform selection
#
# **Input:** `Dataset/DJI_0418.MP4`.
# **Writes:** `Dataset/frames/80fps_timeline/timeline_*.jpg`, `Dataset/manifests/80fps_timeline.jsonl`, `Results/dataset/80fps_extraction_summary.json`, `Dataset/frames/picked_800_uniform/pick_*.jpg`, and `Dataset/manifests/picked_800_uniform.jsonl`.
# **Guarantee:** 80-FPS timestamps are recorded exactly; repeated source frames are documented instead of generated artificially.

# In[3]:


# %% 02B) 80-FPS timeline extraction and exact 800-frame selection
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


# ## Notebook-only implementation — augmentation and grouped split
#
# **Input:** exactly 800 files in `Dataset/frames/picked_800_uniform/`.
# **Writes:** ten variants per image to `Dataset/augmentations/from_800/`, their manifest, the `Dataset/yolo_green_apple/images/{train,val,test}/` split, `Dataset/manifests/grouped_split.jsonl`, and `Dataset/yolo_green_apple/green_apple.yaml`.
# **Split:** each original and its ten variants stay together: 560/80/160 source groups (70/10/20).

# In[4]:


# %% 02C) Augmentation and leakage-safe 70/10/20 split
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


# ## Notebook-only implementation — automatic green-apple labels
#
# **Input:** split images in `Dataset/yolo_green_apple/images/`.
# **Writes:** YOLO labels in `Dataset/yolo_green_apple/labels/{train,val,test}/`, box previews in `Dataset/yolo_green_apple/auto_label_previews/`, `Dataset/manifests/automatic_green_apple_labels.jsonl`, and dataset reports under `Results/dataset/`.
# **Rule:** COCO `apple` proposals become the single project class `green_apple`; no manual-labeling program is used.

# In[5]:


# %% 02D) COCO-assisted automatic labels, previews, and dataset audit
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
        (paths.yolo_labels / split).mkdir(parents=True, exist_ok=True)
        (paths.auto_previews / split).mkdir(parents=True, exist_ok=True)
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
            mean_confidence = float(np.mean([box.score for box in boxes if box.score is not None])) if boxes else 0.0
            all_rows.append({"image": image_path.name, "split": split, "status": "generated", "boxes": len(boxes), "mean_detector_confidence": round(mean_confidence, 6)})
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


# ## Notebook-only implementation — AAM, EBCA, FM, KWConv
#
# **Purpose:** define and integrate the requested AAM → EBCA → FM → KWConv enhancement neck before the YOLOv8 Detect head.
# **Writes when tested:** `Results/architecture/module_smoke_test.json`.
# **Feature flow:** P3/P4/P5 → AAM → EBCA → FM → KWConv → Detect.

# In[6]:


# %% 02E) Custom AAM, EBCA, FM, KWConv modules and enhanced Detect head
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

        @max_det.setter
        def max_det(self, value: int) -> None:
            self.detect.max_det = value

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


# ## Notebook-only implementation — training and measured metrics
#
# **Input:** audited YAML at `Dataset/yolo_green_apple/green_apple.yaml`.
# **Writes:** checkpoints, curves, plots, and `results.csv` under `Results/training/`; validation outputs and `Results/validation/precision_recall_f1_summary.json`.
# **Training behavior:** uses `patience=0`, so configured epochs complete without early stopping.

# In[7]:


# %% 02F) Enhanced YOLOv8 training, validation, precision, recall, F1, and mAP
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


#
# ## Notebook-only implementation — AdaBins depth and duplicate-safe tracking
#
# **Input:** trained weights, the source MP4, and optionally `third_party/AdaBins/`.
# **Writes:** `Results/inference/tracked_green_apples.mp4`, `Results/inference/tracks.csv`, and `Results/inference/tracking_summary.json`.
# **Tracking behavior:** two-stage confidence association uses motion, IoU, optional relative depth, persistent IDs, and one-time line-crossing counts.

# In[8]:


# %% 02G) AdaBins relative depth, depth-aware ByteTrack, and unique count
class AdaBinsDepth:
    """Official AdaBins adapter; output is relative depth without orchard calibration."""

    def __init__(
        self,
        repository: str | Path,
        dataset: str = "kitti",
        device: str | None = None,
        input_size: tuple[int, int] = (928, 512),
    ) -> None:
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
        self.input_size = input_size

    def predict(self, bgr_frame: np.ndarray) -> np.ndarray:
        model_frame = cv2.resize(bgr_frame, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
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


def apparent_scale_corrected_depths(
    detections: Iterable[FruitDetection],
    *,
    reference_confidence: float = 0.50,
    exponent: float = 0.65,
    minimum_factor: float = 0.50,
    maximum_factor: float = 2.50,
) -> list[FruitDetection]:
    """Fuse raw monocular depth with apparent apple size for small orchard fruit.

    The per-frame reference diameter is the median equivalent box diameter
    ``sqrt(width * height)`` among
    detections at or above ``reference_confidence``.  The output remains a
    relative/estimated depth and must not be described as calibrated metres.
    """
    items = list(detections)
    reference_candidates = [
        math.sqrt(max(1e-6, (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        for item in items
        if item.score >= reference_confidence
    ]
    if not reference_candidates:
        reference_candidates = [
            math.sqrt(max(1e-6, (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
            for item in items
        ]
    if not reference_candidates:
        return items
    reference_diameter = float(np.median(reference_candidates))
    corrected: list[FruitDetection] = []
    for item in items:
        if item.depth is None:
            corrected.append(item)
            continue
        diameter = math.sqrt(max(1e-6, (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        factor = float(np.clip((reference_diameter / diameter) ** exponent, minimum_factor, maximum_factor))
        corrected.append(FruitDetection(item.bbox, item.score, item.depth * factor))
    return corrected


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
                detections = apparent_scale_corrected_depths(detections) if adabins is not None else detections
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

# Existing short cells below use fp.function_name, but fp is this notebook module.
import sys
fp = sys.modules[__name__]
paths = fp.project_paths(PROJECT_ROOT)
paths.ensure_layout()
print('Complete implementation loaded from the executable notebook export.')


# ## Detailed project path map
#
# Run the next cell before the workflow cells. It prints every input, generated-data, label, model, metrics, and inference path used by this notebook. The folders may initially be empty; later cells create their contents only when their `RUN_...` flag is enabled.

# In[9]:


# %% 02H) Print all input, dataset, result, and model paths
path_map = {
    'source_video_read_only': paths.raw_video,
    '80fps_timeline': paths.frames_80fps,
    'uniform_800_frames': paths.picked_800,
    'augmentation_images': paths.augmented_800,
    'manifests': paths.manifests,
    'yolo_images': paths.yolo_images,
    'automatic_yolo_labels': paths.yolo_labels,
    'automatic_label_previews': paths.auto_previews,
    'dataset_yaml': paths.data_yaml,
    'hardware_report': paths.hardware_results / 'runtime.json',
    'dataset_reports': paths.dataset_results,
    'architecture_reports': paths.architecture_results,
    'training_runs_and_weights': paths.training_results,
    'validation_metrics': paths.validation_results,
    'tracking_video_and_csv': paths.inference_results,
    'optional_adabins_repository': PROJECT_ROOT / 'third_party' / 'AdaBins',
}
for name, location in path_map.items():
    print(f'{name:30} -> {location}')


# ## Runtime and GPU check
#
# Records installed PyTorch, CUDA availability, NVIDIA driver response, and the selected device. Dataset preparation can use CPU; full YOLOv8x training is protected until a CUDA-enabled PyTorch build and working NVIDIA driver are available.
#
# **Input:** installed NVIDIA driver and the notebook virtual environment.
# **Writes:** `Results/hardware/runtime.json`.
# **Use next:** only enable full training after this reports `cuda_available: true` and `recommended_device: 0`.

# In[10]:


# %% 03) Detect NVIDIA/CUDA and choose the execution device
runtime = fp.hardware_report(paths)
print(json.dumps(runtime, indent=2))
DEVICE = runtime['recommended_device']
if DEVICE == 'cpu':
    print('CUDA is not usable. Prepare data on CPU, then repair CUDA before full YOLOv8x training.')
    for step in runtime.get('cuda_remediation', []):
        print(' -', step)
    if not runtime.get('nvidia_kernel_module_present', False):
        print('Ubuntu driver repair for this machine:')
        print('sudo apt update && sudo apt install linux-generic-hwe-24.04 nvidia-driver-595-open linux-modules-nvidia-595-open-generic-hwe-24.04')
        print('Then reboot and run nvidia-smi.')
    elif runtime.get('nvidia_smi_returncode', 0) != 0:
        print('The NVIDIA module exists but nvidia-smi still fails. Reboot after the driver upgrade, then rerun this cell.')
    else:
        print('The driver is available; replace the CPU-only PyTorch package with a CUDA-enabled build in .venv.')
else:
    print(f'Using GPU {DEVICE}: {runtime.get("cuda_device")}')


# ## Video preflight and 80-FPS accounting
#
# Inspects the source without writing frames. It reports real frame rate, dimensions, duration, source count, and requested 80-FPS timeline length. This video produces 5,105 output timestamps over about 63.814 seconds.
#
# **Reads:** `Dataset/DJI_0418.MP4` only.
# **Writes:** nothing.
# **Expected report:** 3,825 source frames at 59.940 FPS; 5,105 requested 80-FPS timestamps.

# In[11]:


# %% 04) Inspect the video and calculate the 80-FPS timeline
TARGET_FPS = 80.0
preflight = fp.video_preflight(paths.raw_video, target_fps=TARGET_FPS)
print(json.dumps(preflight, indent=2))
assert preflight['playable'], preflight.get('error', 'Input video is not playable.')
assert preflight['requested_80fps_timeline_frames'] == 5105


# ## Stage 1 — documented 80-FPS extraction
#
# Writes 5,105 frames into `Dataset/frames/80fps_timeline/` and an `80fps_timeline.jsonl` manifest. The MP4 is read-only; only generated `timeline_*.jpg` files are replaceable when rebuild is enabled.
#
# **Reads:** `Dataset/DJI_0418.MP4`.
# **Writes:** `Dataset/frames/80fps_timeline/` and `Dataset/manifests/80fps_timeline.jsonl`.
# **Expected count:** 5,105 images.

# In[12]:


# %% 05) Extract the 80-FPS timeline
RUN_80FPS_EXTRACTION = False  # COMPLETED: 5,105 timeline images exist; set True only to regenerate.
REBUILD_80FPS_TIMELINE = False

if RUN_80FPS_EXTRACTION:
    timeline_manifest = fp.extract_80fps_timeline(paths, target_fps=TARGET_FPS, overwrite=REBUILD_80FPS_TIMELINE)
    print(f'80-FPS timeline frames written: {len(timeline_manifest):,}')
    print('Manifest:', paths.manifests / '80fps_timeline.jsonl')
else:
    print('STEP 03 READY — set RUN_80FPS_EXTRACTION=True to write 5,105 images to Dataset/frames/80fps_timeline/.')


# ## Stage 2 — exact uniform sample of 800
#
# Selects exactly 800 evenly spaced timeline positions and copies them to `Dataset/frames/picked_800_uniform/`. A manifest preserves the selected timeline position and filename for every picked image.
#
# **Reads:** 5,105 `timeline_*.jpg` images.
# **Writes:** exactly 800 `pick_*.jpg` images in `Dataset/frames/picked_800_uniform/` and their manifest.

# In[13]:


# %% 06) Uniformly select 800 frames
RUN_UNIFORM_SELECTION = False  # COMPLETED: 800 uniform images exist; set True only to regenerate.
REBUILD_UNIFORM_SELECTION = False

if RUN_UNIFORM_SELECTION:
    picked_manifest = fp.pick_uniform_800(paths, count=800, overwrite=REBUILD_UNIFORM_SELECTION)
    print(f'Uniform frames picked: {len(picked_manifest):,}')
    for row in picked_manifest[:5]:
        print(row)
else:
    print('STEP 04 WAITING — requires Dataset/frames/80fps_timeline/timeline_*.jpg from STEP 03.')


# ## Stage 3 — augment the 800 picked images
#
# Creates ten documented visual variants per selected image: flips, brightness changes, rotations, small rotation, motion blur, and Gaussian noise. It creates no labels at this stage because automatic labeling is intentionally performed after splitting.
#
# **Reads:** 800 picked images.
# **Writes:** 8,000 augmentation images in `Dataset/augmentations/from_800/`, plus `Results/dataset/augmentation_summary.json`.

# In[14]:


# %% 07) Create 8,000 augmentation images
RUN_AUGMENTATION = False  # COMPLETED: 8,000 variants exist; set True only to regenerate.
REBUILD_AUGMENTATION = False

if RUN_AUGMENTATION:
    augmentation_summary = fp.augment_picked_800(paths, seed=42, overwrite=REBUILD_AUGMENTATION)
    print(json.dumps(augmentation_summary, indent=2))
else:
    print('STEP 05 WAITING — requires 800 images in Dataset/frames/picked_800_uniform/.')


# ## Dataset inventory
#
# Reports the exact requested counts of 80-FPS timeline images, uniform 800 selection, augmentations, and their combined total. This is read-only reporting after the generation stages.
#
# **Reads:** generated dataset folders and manifests only.
# **Writes:** nothing.
# **Expected total after augmentation:** 800 originals + 8,000 variants = 8,800 images.

# In[15]:


# %% 08) Display all frame counts and manifest locations
def count_images(folder, pattern='*.jpg'):
    return len(list(Path(folder).glob(pattern)))

inventory = {
    '80fps_timeline_frames': count_images(paths.frames_80fps, 'timeline_*.jpg'),
    'uniformly_picked_frames': count_images(paths.picked_800, 'pick_*.jpg'),
    'augmentation_frames': count_images(paths.augmented_800, '*__*.jpg'),
}
inventory['picked_plus_augmentations'] = inventory['uniformly_picked_frames'] + inventory['augmentation_frames']
print(json.dumps(inventory, indent=2))
print('Picked manifest      :', paths.manifests / 'picked_800_uniform.jsonl')
print('Augmentation manifest:', paths.manifests / 'augmentation_manifest.jsonl')
if inventory['80fps_timeline_frames'] == 0:
    print('\nSTEP 06 WAITING — no generated frames exist yet. Go to STEP 03, set RUN_80FPS_EXTRACTION=True, and run that cell first.')
elif inventory['uniformly_picked_frames'] == 0:
    print('\nSTEP 06 WAITING — timeline exists. Continue with STEP 04 to create exactly 800 uniform frames.')
elif inventory['augmentation_frames'] == 0:
    print('\nSTEP 06 WAITING — 800 frames exist. Continue with STEP 05 to create 8,000 augmentation frames.')
else:
    print('\nSTEP 06 COMPLETE — expected total is 8,800 images: 800 originals + 8,000 augmentations.')


# ## Stage 4 — grouped 70/10/20 split
#
# This split preserves every original and its ten derivatives in one partition, preventing augmentation leakage. It produces exactly 560/80/160 source-frame groups and 6,160/880/1,760 images in train/validation/test. Labeling occurs only after this split.
#
# **Reads:** the 800 originals and 8,000 variants.
# **Writes:** 6,160 train, 880 validation, and 1,760 test images under `Dataset/yolo_green_apple/images/`.

# In[16]:


# %% 09) Build the leakage-safe YOLO image split
RUN_GROUPED_SPLIT = False  # COMPLETED: 70/10/20 grouped split exists; set True only to rebuild.
REBUILD_GROUPED_SPLIT = False

if RUN_GROUPED_SPLIT:
    split_summary = fp.build_grouped_dataset_split(paths, train_fraction=0.70, val_fraction=0.10, overwrite=REBUILD_GROUPED_SPLIT)
    print(json.dumps(split_summary, indent=2))
    assert (split_summary['train_groups'], split_summary['val_groups'], split_summary['test_groups']) == (560, 80, 160)
else:
    print('STEP 07 WAITING — requires 800 picked images and 8,000 augmentation images.')


# ## Stage 5 — automatic single-class green-apple labels
#
# No manual labels are used. A COCO-pretrained YOLOv8 detector proposes the COCO `apple` category from full-frame and overlapping tiled inference. NMS removes duplicate detections and every accepted box is saved as class `green_apple`; previews and an audit manifest are generated. These are pseudo-labels, so they cannot prove a real >90% result without independent ground-truth test labels.
#
# **Reads:** all final split images.
# **Writes:** matching YOLO `.txt` files, previews, and `Results/dataset/automatic_label_summary.json`.
# **Important:** no human annotation GUI is used.

# In[17]:


# %% 10) Automatically label all split images
RUN_AUTOMATIC_LABELING = False  # COMPLETED: 8,800 automatic labels/previews exist; set True only to regenerate.
REBUILD_AUTOMATIC_LABELS = False
AUTO_LABEL_GPU_GIB = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if DEVICE != 'cpu' else 0.0
AUTO_LABEL_LOW_VRAM = AUTO_LABEL_GPU_GIB <= 6.0
AUTO_LABEL_MODEL = 'yolov8s.pt' if AUTO_LABEL_LOW_VRAM else 'yolov8x.pt'
AUTO_LABEL_CONFIDENCE = 0.15
AUTO_LABEL_IMAGE_SIZE = 960 if AUTO_LABEL_LOW_VRAM else 1280
print(f'Auto-label profile: {AUTO_LABEL_MODEL} at {AUTO_LABEL_IMAGE_SIZE}px on {AUTO_LABEL_GPU_GIB:.1f} GiB GPU memory.')

if RUN_AUTOMATIC_LABELING:
    auto_label_summary = fp.auto_label_after_split(
        paths, model_name=AUTO_LABEL_MODEL, confidence=AUTO_LABEL_CONFIDENCE,
        image_size=AUTO_LABEL_IMAGE_SIZE, device=DEVICE, tiled=True,
        overwrite=REBUILD_AUTOMATIC_LABELS,
    )
    print(json.dumps(auto_label_summary, indent=2))
    print('Labels  :', paths.yolo_labels)
    print('Previews:', paths.auto_previews)
else:
    print('STEP 08 WAITING — set RUN_AUTOMATIC_LABELING=True after STEP 07; labels and previews will be written under Dataset/yolo_green_apple/.')


# ## Dataset and label audit
#
# Verifies image/label parity and reports the boxes, empty labels, and the one permitted class. It saves `Results/dataset/dataset_audit.json`. Training refuses to start until this audit passes.
#
# **Reads:** `Dataset/yolo_green_apple/images/` and `labels/`.
# **Writes:** `Results/dataset/dataset_audit.json`.
# **Required before training:** every image must have one label file.

# In[18]:


# %% 11) Audit automatic labels
audit = fp.validate_dataset(paths)
print(json.dumps(audit, indent=2))
if any(part['images'] for part in audit['splits'].values()):
    assert audit['valid'], 'Every split image must have an automatic YOLO label file before training.'
else:
    print('Dataset split and automatic labels have not been created yet; run Cells 04–09 before training.')


# ## AAM — Attribute Attention Module
#
# **Purpose:** improve green-apple representation using three parallel attention branches.
# **Code cell:** `02E`, class `AAM`.
# **Input:** P4 feature tensor `F`.  **Operations:** colour branch uses pooled channel attention; shape branch uses depthwise 3×3 convolution; scale branch uses averaged-context 1×1 convolution; learned softmax weights fuse the three maps.
# **Output:** `F ⊙ sigmoid(αF_colour + βF_shape + γF_scale)` with the same P4 shape.

# ## EBCA — Efficient Bidirectional Cross-Attention
#
# **Purpose:** handle overlapping fruit by exchanging high-resolution P3 information and deeper P4 semantics.
# **Code cell:** `02E`, class `EBCA`.
# **Input:** P3 and AAM-enhanced P4.  **Operations:** project/upsample P4 to gate P3, project/pool P3 to gate P4, use sigmoid attention and normalization.
# **Output:** enhanced P3 and enhanced P4 with their original spatial sizes.

# ## FM — Focal Modulation
#
# **Purpose:** suppress background/occluded context while retaining visible apple information.
# **Code cell:** `02E`, class `FM`.
# **Input:** EBCA-enhanced P4.  **Operations:** query projection, three hierarchical depthwise contextual convolutions, learned gates, global context, and modulation.
# **Output:** context-aware P4 used by KWConv.

# ## KWConv — Kernel Warehouse Convolution
#
# **Purpose:** make the neck receptive field adaptive for small and large apples.
# **Code cell:** `02E`, class `KWConv`.
# **Input:** P3, FM-enhanced P4, and P5.  **Operations:** global pooling selects soft weights across 3×3, 5×5, and 7×7 convolution branches; their weighted sum is normalized and activated.
# **Output:** adaptive P3/P4/P5 for the standard YOLO Detect head.

# ## AdaBins + relative position processing
#
# **Purpose:** add per-box relative depth so nearby/foreground and distant/background apples can be distinguished.
# **Code cells:** `02G` and workflow Cell 16.
# **Input:** RGB frame and detected box `(x1,y1,x2,y2)`.  **Operations:** AdaBins produces a depth map; the robust trimmed median inside each box produces an object-level relative depth.
# **Output:** detection tuple `(box, confidence, relative_depth)`. Depth is relative until the orchard camera is calibrated.

# ## Depth-aware ByteTrack and unique count
#
# **Purpose:** preserve one fruit ID across frames and count it only once.
# **Code cells:** `02G` and workflow Cell 17.
# **Input:** high/low-confidence apple detections plus optional relative depth.  **Operations:** predict tracks, associate using Hungarian cost from IoU, score, and depth; update matched tracks; expire lost tracks; count an ID once after line crossing.
# **Output paths:** `Results/inference/tracked_green_apples.mp4`, `tracks.csv`, and `tracking_summary.json`.

# ## AAM, EBCA, FM, and KWConv
#
# The four implemented PyTorch modules match the submitted architecture: AAM combines colour/shape/scale attention, EBCA exchanges P3/P4 information, FM gates contextual features for occlusion, and KWConv selects mixed receptive fields for scale variation.
#
# **Reads:** no data files.
# **Writes:** nothing in this inspection cell.
# **Purpose:** demonstrate each custom tensor module before the integrated smoke test.

# In[19]:


# %% 12) Construct and inspect all custom feature modules
import torch

aam = fp.AAM(channels=128).to(DEVICE).eval()
ebca = fp.EBCA(p3_channels=64, p4_channels=128).to(DEVICE).eval()
fm = fp.FM(channels=128).to(DEVICE).eval()
kwconv = fp.KWConv(channels=128).to(DEVICE).eval()
with torch.inference_mode():
    p3 = torch.zeros(1, 64, 80, 80, device=DEVICE)
    p4 = torch.zeros(1, 128, 40, 40, device=DEVICE)
    print('AAM   :', tuple(aam(p4).shape))
    print('EBCA  :', tuple(shape.shape for shape in ebca(p3, p4)))
    print('FM    :', tuple(fm(p4).shape))
    print('KWConv:', tuple(kwconv(p4).shape))


# ## Integrated enhanced YOLOv8 feature neck
#
# The proposed neck runs **AAM → EBCA → FM → KWConv** on YOLO P3/P4/P5 before the standard Detect head. The smoke test confirms compatible tensor shapes and writes an architecture record to `Results/architecture/`.
#
# **Reads:** no dataset files.
# **Writes:** `Results/architecture/module_smoke_test.json`.
# **Checks:** P3/P4/P5 output tensor shapes remain compatible with YOLOv8 Detect.

# In[20]:


# %% 13) Verify the integrated AAM–EBCA–FM–KWConv neck
architecture_report = fp.module_smoke_test(paths, device=DEVICE)
print(json.dumps(architecture_report, indent=2))


# ## Full training run
#
# This uses one-class enhanced YOLOv8 with auto GPU batch sizing and 100 epochs. The custom trainer inserts the enhanced neck into the model actually being optimized. Results, checkpoints, plots, and curves go to `Results/training/`. `patience=0` prevents an early-stopping interruption.
#
# **Reads:** `Dataset/yolo_green_apple/green_apple.yaml` and automatic labels.
# **Writes:** `Results/training/aam_ebca_fm_kwconv_green/`, including `weights/best.pt`, `weights/last.pt`, curves, plots, and `results.csv`.
# **RTX 3050 Ti 4-GB profile:** the cell automatically selects YOLOv8s at 640 pixels to fit this GPU. The custom AAM/EBCA/FM/KWConv neck remains the same. A GPU with more than 8 GB uses the YOLOv8x/1,024-pixel profile.

# In[ ]:


# %% 14) Train the VRAM-aware enhanced YOLOv8 model through all requested epochs
RUN_FULL_TRAINING = False  # STEP 11: set True only when STEP 09 audit is valid and CUDA is available.
REQUIRE_CUDA_FOR_TRAINING = True
GPU_MEMORY_GIB = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if DEVICE != 'cpu' else 0.0
LOW_VRAM_PROFILE = GPU_MEMORY_GIB <= 6.0
TRAINING_MODEL = 'yolov8s.pt' if LOW_VRAM_PROFILE else 'yolov8x.pt'
TRAINING_EPOCHS = 300
TRAINING_IMAGE_SIZE = 640 if LOW_VRAM_PROFILE else 1024
TRAINING_BATCH = 1  # Explicit safe batch size for the 4-GB RTX 3050 Ti; avoids auto-batch memory probing.
TRAINING_WORKERS = 4
TRAINING_NAME = 'aam_ebca_fm_kwconv_green'
print(f'GPU memory: {GPU_MEMORY_GIB:.1f} GiB | model: {TRAINING_MODEL} | image size: {TRAINING_IMAGE_SIZE}')

if RUN_FULL_TRAINING:
    if REQUIRE_CUDA_FOR_TRAINING and DEVICE == 'cpu':
        raise RuntimeError('CUDA is unavailable. Install a working NVIDIA driver and CUDA-enabled PyTorch, rerun Cell 02, then start the complete run.')
    train_results = fp.train_enhanced_yolov8(
        paths, model_name=TRAINING_MODEL, epochs=TRAINING_EPOCHS,
        image_size=TRAINING_IMAGE_SIZE, batch=TRAINING_BATCH, device=DEVICE,
        workers=TRAINING_WORKERS, run_name=TRAINING_NAME,
    )
    print('Training completed:', train_results)
else:
    print('STEP 11 WAITING — set RUN_FULL_TRAINING=True to run all 300 epochs and write Results/training/aam_ebca_fm_kwconv_green/.')


# ## Precision, recall, F1, and mAP
#
# After training, validation generates Ultralytics metrics and a compact final report with precision, recall, calculated F1, mAP@50, and mAP@50–95 in `Results/validation/precision_recall_f1_summary.json`. The score is scientifically valid only against independent ground truth, not test pseudo-labels created by the same COCO model.
#
# **Reads:** `Results/training/.../weights/best.pt` and final run CSV.
# **Writes:** validation plots and `Results/validation/precision_recall_f1_summary.json`.

# In[22]:


# %% 15) Validate trained weights and save final metrics
RUN_FINAL_VALIDATION = False  # STEP 12: requires Results/training/.../weights/best.pt.
BEST_WEIGHTS = paths.training_results / TRAINING_NAME / 'weights' / 'best.pt'

if RUN_FINAL_VALIDATION:
    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f'Missing trained weights: {BEST_WEIGHTS}')
    validation_results = fp.validate_model(paths, BEST_WEIGHTS, image_size=TRAINING_IMAGE_SIZE, device=DEVICE)
    metric_summary = fp.summarize_metrics(paths, paths.training_results / TRAINING_NAME)
    print(json.dumps(metric_summary, indent=2))
else:
    print('STEP 12 WAITING — training must create:', BEST_WEIGHTS)


# ## AdaBins relative depth
#
# The optional official AdaBins adapter estimates relative pixel depth. Robust depth per apple box is used to distinguish closer/farther fruit during tracking. Values are not physical metres until camera calibration is supplied.
#
# **Reads:** optional local `third_party/AdaBins/` repository and checkpoint.
# **Writes:** no independent dataset files; depth is consumed by the tracking cell.

# In[23]:


# %% 16) Initialize AdaBins for relative-depth estimation
RUN_ADABINS = False  # STEP 13A: optional; requires the local AdaBins repository/checkpoint.
ADABINS_REPOSITORY = PROJECT_ROOT / 'third_party' / 'AdaBins'
adabins = None

if RUN_ADABINS:
    adabins = fp.AdaBinsDepth(ADABINS_REPOSITORY, dataset='kitti', device=str(DEVICE))
    print('AdaBins ready; tracking will use relative depth.')
else:
    print('Optional stage not initialized. Tracking can run without depth.')


# ## Depth-aware ByteTrack-style tracking and duplicate-safe count
#
# The final stage combines detections with a two-stage confidence association, motion prediction, IoU, and optional depth. Persistent IDs are written frame-by-frame, while a line counter records each confirmed track only once. It outputs annotated video, tracks CSV, and summary in `Results/inference/`.
#
# **Reads:** source MP4, trained `best.pt`, and optionally AdaBins.
# **Writes:** annotated MP4, track CSV, and tracking JSON under `Results/inference/`.

# In[24]:


# %% 17) Run tracking and save the unique apple count
RUN_TRACKING = False  # STEP 13B: requires best.pt; AdaBins is optional but adds relative depth.
COUNT_LINE = ((0, 540), (1919, 540))

if RUN_TRACKING:
    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f'Missing trained weights: {BEST_WEIGHTS}')
    tracking_summary = fp.run_depth_aware_tracking(
        paths, weights=BEST_WEIGHTS, adabins=adabins, confidence=0.10,
        image_size=TRAINING_IMAGE_SIZE, device=DEVICE, line=COUNT_LINE,
    )
    print(json.dumps(tracking_summary, indent=2))
else:
    print('STEP 13B WAITING — requires trained best.pt; outputs will be written to Results/inference/.')


# ## Final artifact inventory
#
# Lists the generated local data and results after running the relevant cells. The repository contains only the reusable pipeline, notebook, environment configuration, and documentation—not the video, generated images, labels, checkpoints, or results.
#
# **Reads:** all generated `Dataset/` and `Results/` folders.
# **Writes:** nothing.
# **Use:** final confirmation of every file produced by the workflow.

# In[25]:


# %% 18) List dataset, results, manifests, and metrics
for folder in (
    paths.frames_80fps, paths.picked_800, paths.augmented_800, paths.yolo,
    paths.dataset_results, paths.architecture_results, paths.training_results,
    paths.validation_results, paths.inference_results,
):
    print(f'{folder.relative_to(paths.root)}: {sum(p.is_file() for p in folder.rglob("*")):,} files')

print('\nReports:')
for report in sorted(paths.manifests.glob('*.jsonl')) + sorted(paths.dataset_results.glob('*.json')) + sorted(paths.validation_results.glob('*.json')):
    print(' -', report.relative_to(paths.root))
