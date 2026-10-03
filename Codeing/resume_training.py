#!/usr/bin/env python3
"""Safely resume the enhanced green-apple YOLO run from its last checkpoint.

The checkpoint stores notebook-defined classes under ``__main__``. This launcher
loads the exact compatibility definitions from the primary notebook, rebuilds
the enhanced head before loading checkpoint tensors, and refuses to train unless
every saved model tensor is restored exactly.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "Codeing" / "AAM, EBCA, FM, KWConv 100.ipynb"
RUN_DIR = PROJECT_ROOT / "Results" / "training" / "aam_ebca_fm_kwconv_green"
LAST_WEIGHTS = RUN_DIR / "weights" / "last.pt"
LOCK_PATH = RUN_DIR / ".resume.lock"


def load_notebook_definitions() -> None:
    """Define the exact notebook classes required to unpickle the checkpoint."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markers = (
        "# %% 02A) Core paths, manifests, and video metadata",
        "# %% 02E) Custom AAM, EBCA, FM, KWConv modules and enhanced Detect head",
    )
    for marker in markers:
        matches = [
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).startswith(marker)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one notebook cell beginning with: {marker}")
        source = "".join(matches[0]["source"])
        exec(compile(source, f"{NOTEBOOK.name}:{marker}", "exec"), globals(), globals())


def transfer_and_verify(model: Any, weights: Any, *, verbose: bool) -> Any:
    """Load all checkpoint tensors and fail closed on any mismatch."""
    import torch

    source_state = weights.float().state_dict()
    model.load(weights, verbose=verbose)
    target_state = model.state_dict()

    missing = sorted(set(source_state) - set(target_state))
    unexpected = sorted(set(target_state) - set(source_state))
    shape_mismatches = sorted(
        key for key in source_state.keys() & target_state.keys() if source_state[key].shape != target_state[key].shape
    )
    value_mismatches = sorted(
        key
        for key in source_state.keys() & target_state.keys()
        if source_state[key].shape == target_state[key].shape
        and not torch.equal(source_state[key].detach().cpu(), target_state[key].detach().cpu())
    )
    if missing or unexpected or shape_mismatches or value_mismatches:
        raise RuntimeError(
            "Checkpoint transfer verification failed: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatches={len(shape_mismatches)}, value_mismatches={len(value_mismatches)}"
        )
    if not isinstance(model.model[-1], EnhancedDetectHead):
        raise TypeError(f"Expected EnhancedDetectHead, found {type(model.model[-1]).__name__}")
    print(f"Verified exact checkpoint transfer: {len(source_state)}/{len(source_state)} tensors", flush=True)
    return model


def checkpoint_summary() -> tuple[Any, dict[str, Any], int, int]:
    """Load and validate the resumable checkpoint metadata."""
    from ultralytics import YOLO

    if not LAST_WEIGHTS.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {LAST_WEIGHTS}")
    resume_model = YOLO(str(LAST_WEIGHTS), task="detect")
    checkpoint = resume_model.ckpt
    if not checkpoint:
        raise RuntimeError("Checkpoint metadata is missing.")

    completed = int(checkpoint.get("epoch", -1)) + 1
    target = int(checkpoint.get("train_args", {}).get("epochs", 0))
    if completed <= 0 or target <= 0:
        raise RuntimeError(f"Invalid checkpoint epoch metadata: completed={completed}, target={target}")
    if checkpoint.get("optimizer") is None or not checkpoint.get("scaler") or checkpoint.get("ema") is None:
        raise RuntimeError("Checkpoint is missing optimizer, scaler, or EMA state required for a true resume.")
    if not isinstance(resume_model.model.model[-1], EnhancedDetectHead):
        raise TypeError("The checkpoint does not contain the enhanced AAM/EBCA/FM/KWConv detection head.")

    data_path = Path(str(checkpoint["train_args"].get("data", "")))
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data configuration is missing: {data_path}")
    print(
        f"Checkpoint ready: {completed}/{target} epochs completed; "
        f"next epoch={completed + 1}; file={LAST_WEIGHTS}",
        flush=True,
    )
    return resume_model, checkpoint, completed, target


def verify_rebuild(resume_model: Any) -> None:
    """Dry-run the same architecture rebuild and transfer used by the trainer."""
    from ultralytics.nn.tasks import DetectionModel

    source = resume_model.model
    channels = int(source.yaml.get("channels", 3))
    rebuilt = DetectionModel(source.yaml, ch=channels, nc=source.nc, verbose=False)
    patch_yolov8_detection_head(rebuilt)
    transfer_and_verify(rebuilt, source, verbose=False)


def acquire_lock() -> Any:
    """Prevent accidental concurrent trainers from using the same run directory."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another resume process already holds {LOCK_PATH}") from error
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="validate the checkpoint without starting training")
    args = parser.parse_args()

    lock_handle = acquire_lock()
    load_notebook_definitions()
    resume_model, _, completed, target = checkpoint_summary()
    if completed >= target:
        print(f"Training is already complete ({completed}/{target} epochs).", flush=True)
        return 0

    verify_rebuild(resume_model)
    if args.check_only:
        print("Resume preflight passed; no training was started.", flush=True)
        return 0

    from ultralytics.models.yolo.detect import DetectionTrainer

    class GreenAppleEnhancedResumeTrainer(DetectionTrainer):
        """Rebuild the custom head before restoring every saved tensor."""

        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            patch_yolov8_detection_head(model)
            if weights is not None:
                transfer_and_verify(model, weights, verbose=verbose)
            return model

    print(f"Starting guarded resume at epoch {completed + 1} of {target}.", flush=True)
    resume_model.train(trainer=GreenAppleEnhancedResumeTrainer, resume=True)
    print("Training completed successfully.", flush=True)
    del lock_handle
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
