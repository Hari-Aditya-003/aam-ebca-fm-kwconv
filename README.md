# AAM, EBCA, FM, KWConv

Cell-by-cell research workflow for green-apple detection, relative-depth localization, and duplicate-safe orchard counting.

## Completed run

Training completed all **300 epochs**. The selected `best.pt` checkpoint corresponds to epoch 24 and was revalidated on 880 validation images.

| Metric | Best-checkpoint validation |
|---|---:|
| Precision | 0.80431 |
| Recall | 0.71415 |
| F1 | 0.75649 |
| mAP@0.50 | 0.80977 |
| mAP@0.50:0.95 | 0.58741 |

These are provisional metrics against automatically generated pseudo-labels, not independent human ground truth. See the [results summary](Results/README.md), [completed status](Results/status/live_training_status.json), and [per-epoch metrics](Results/training/aam_ebca_fm_kwconv_green/results.csv).

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
  resume_training.py                # guarded exact-checkpoint resume launcher
  render_result_videos.py           # rebuild final videos from published CSVs
  export_max5m_videos_and_frames.py # filtered videos and local annotated frames
  AAM, EBCA, FM, KWConv 100 (5) (3).ipynb  # supplied reference notebook
  requirements.txt
Dataset/                            # local-only: video, frames, images, labels, manifests
Results/                            # reports plus LFS-hosted final videos and checkpoints
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

`Dataset/` remains local because the generated dataset is approximately 36 GB. Final videos, prediction tables, and detector checkpoints are published through Git LFS. Generated frames, labels, dataset previews, and the third-party AdaBins repository/checkpoint remain excluded. Clone with Git LFS enabled to download the large result artifacts.
