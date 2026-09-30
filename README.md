# AAM, EBCA, FM, KWConv

Cell-by-cell research workflow for green-apple detection, relative-depth localization, and duplicate-safe orchard counting.

## Current training snapshot

Training was active when this snapshot was recorded on **30 September 2026 at 20:56 IST**. The run has completed **65 of 300 epochs (21.7%)**.

| Metric | Latest (epoch 65) | Best observed | Best epoch |
|---|---:|---:|---:|
| Precision | 0.80084 | 0.82107 | 36 |
| Recall | 0.71182 | 0.71746 | 48 |
| mAP@0.50 | 0.79950 | 0.81440 | 28 |
| mAP@0.50:0.95 | 0.57659 | 0.58744 | 24 |

These values are provisional validation metrics against automatically generated pseudo-labels; they are not final independent-ground-truth results. See the [results summary](Results/README.md), [live-status snapshot](Results/status/live_training_status.json), and [per-epoch metrics](Results/training/aam_ebca_fm_kwconv_green/results.csv).

## Objectives

- Detect small, overlapping, and partially occluded green apples with an enhanced YOLOv8 neck: Attribute Attention Module (AAM), Efficient Bidirectional Cross-Attention (EBCA), Focal Modulation (FM), and Kernel Warehouse Convolution (KWConv).
- Estimate relative pixel depth with AdaBins to separate foreground/background fruit and support spatial association.
- Use a depth-aware, two-stage ByteTrack-style tracker to maintain fruit IDs and avoid duplicate counts across video frames.

## Notebook workflow

Open [AAM, EBCA, FM, KWConv 100.ipynb](<Codeing/AAM, EBCA, FM, KWConv 100.ipynb>) and run cells in order. Use [AAM, EBCA, FM, KWConv 100 - Results and Status.ipynb](<Codeing/AAM, EBCA, FM, KWConv 100 - Results and Status.ipynb>) to inspect progress and generate final result artifacts without changing the training workflow.

1. Check PyTorch, CUDA, and NVIDIA-driver availability.
2. Inspect `Dataset/DJI_0418.MP4`: 3,825 source frames at 59.940 FPS, 1,920×1,080.
3. Create a timestamp-accurate 80-FPS timeline: **5,105 frames**. As the source is below 80 FPS, repeated source-frame indices are documented in the manifest rather than producing artificial camera frames.
4. Uniformly select exactly **800** frames.
5. Create **10 augmentations per selected frame**: 8,000 variants, 8,800 images in total.
6. Keep each original image and all its variants together in the grouped 70/10/20 split: 560/80/160 source groups and 6,160/880/1,760 train/validation/test images.
7. After splitting, create single-class `green_apple` YOLO labels automatically from COCO-pretrained YOLOv8 apple proposals using full-frame and overlapping-tile inference.
8. Audit labels, construct/test AAM, EBCA, FM, and KWConv, then train the enhanced YOLOv8 model for all configured epochs (`patience=0`).
9. Save precision, recall, F1, mAP, plots, checkpoints, relative-depth outputs, IDs, tracked video, and unique fruit count under `Results/`.

The `RUN_...` flags make expensive or file-writing stages explicit. They are `False` initially; set them to `True` only when ready to execute that stage. The source video is never changed.

## Layout

```text
Codeing/
  AAM, EBCA, FM, KWConv 100.ipynb  # documented research notebook
  AAM, EBCA, FM, KWConv 100 - Results and Status.ipynb  # monitoring and result export
  AAM, EBCA, FM, KWConv 100 (5) (3).ipynb  # supplied reference notebook
  requirements.txt
Dataset/                            # local-only: video, frames, images, labels, manifests
Results/                            # curated reports tracked; large/generated artifacts local-only
third_party/AdaBins/                # local AdaBins source/checkpoint (not committed)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r Codeing/requirements.txt
.venv/bin/python -m ipykernel install --user --name fruitvision --display-name "FruitVision (.venv)"
```

For full training, use an NVIDIA driver and a CUDA-enabled PyTorch build. Cell 02 records the actual runtime state; it does not assume that a visible GPU is usable.

## Evaluation note

Automatic COCO-derived labels are pseudo-labels. They enable a fully automatic dataset-building experiment, but they cannot establish a scientifically valid claim such as “above 90% precision” when the held-out labels originate from the same detector. For a defensible precision/recall/F1 result, evaluate the final model on independently created ground-truth annotations. The notebook still saves all model-output metrics and makes this distinction explicit.

## Data and result policy

`Dataset/` is always local and is never committed. The repository includes only curated aggregate results such as status JSON, per-epoch metrics, runtime information, module smoke-test output, and final plots when available. Checkpoints, videos, generated frames, labels, dataset-derived previews, and third-party AdaBins weights remain ignored. The complete implementation and workflow are contained in the notebooks; no project `.py` source file is required.
