# AAM, EBCA, FM, KWConv

[![GitHub stars](https://img.shields.io/github/stars/Hari-Aditya-003/aam-ebca-fm-kwconv?style=flat-square)](https://github.com/Hari-Aditya-003/aam-ebca-fm-kwconv/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Hari-Aditya-003/aam-ebca-fm-kwconv?style=flat-square)](https://github.com/Hari-Aditya-003/aam-ebca-fm-kwconv/forks)
[![Default branch](https://img.shields.io/badge/branch-main-2563eb?style=flat-square)](https://github.com/Hari-Aditya-003/aam-ebca-fm-kwconv/tree/main)

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

For a line-addressable Python version of the complete workflow, use [`Codeing/aam_ebca_fm_kwconv_main.py`](Codeing/aam_ebca_fm_kwconv_main.py). The formula-to-code audit is available as [Markdown](Documentation/AAM_EBCA_FM_KWConv_Technical_Documentation.md) and [DOCX](Documentation/AAM_EBCA_FM_KWConv_Technical_Documentation.docx).

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
  aam_ebca_fm_kwconv_main.py        # complete line-addressable Python source
  AAM, EBCA, FM, KWConv 100 - Results and Status.ipynb  # monitoring and result export
  build_technical_document.py       # rebuild the DOCX from Markdown
  resume_training.py                # guarded exact-checkpoint resume launcher
  render_result_videos.py           # rebuild final videos from published CSVs
  export_max5m_videos_and_frames.py # filtered videos and local annotated frames
  export_all_apple_confidence_video.py # all detections with confidence only
  AAM, EBCA, FM, KWConv 100 (5) (3).ipynb  # supplied reference notebook
  requirements.txt
Dataset/                            # local-only: video, frames, images, labels, manifests
Results/                            # reports plus LFS-hosted final videos and checkpoints
Documentation/                      # formula-to-code technical document (MD and DOCX)
third_party/AdaBins/                # local AdaBins source/checkpoint (not committed)
```

## Setup

Clone with Git LFS before installing Python dependencies:

```bash
git lfs install
git clone https://github.com/Hari-Aditya-003/aam-ebca-fm-kwconv.git
cd aam-ebca-fm-kwconv
git lfs pull
```

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r Codeing/requirements.txt
fruitvision_kernel="fruitvision-$(.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
.venv/bin/python -m ipykernel install --user --name "$fruitvision_kernel" --display-name "FruitVision ($fruitvision_kernel)"
```

For full training, use an NVIDIA driver and a CUDA-enabled PyTorch build. Cell 02 records the actual runtime state; it does not assume that a visible GPU is usable.

## Evaluation note

Automatic COCO-derived labels are pseudo-labels. They enable a fully automatic dataset-building experiment, but they cannot establish a scientifically valid claim such as “above 90% precision” when the held-out labels originate from the same detector. For a defensible precision/recall/F1 result, evaluate the final model on independently created ground-truth annotations. The notebook still saves all model-output metrics and makes this distinction explicit.

## Data and result policy

`Dataset/` remains local because the generated dataset is approximately 36 GB. Final videos, prediction tables, and detector checkpoints are published through Git LFS. Generated frames, labels, dataset previews, and the third-party AdaBins repository/checkpoint remain excluded. Clone with Git LFS enabled to download the large result artifacts.

## Documentation and contributions

- [Deployment and operation](docs/DEPLOYMENT.md)
- [Contributing](CONTRIBUTING.md): setup, checks, issue reports, and pull requests
- [Branches](https://github.com/Hari-Aditya-003/aam-ebca-fm-kwconv/branches): `main` is the default branch; use a focused branch for each change.

If this project helps you, consider starring it. Fork it to propose improvements, then open a pull request against `main`. The stars and forks badges above update from GitHub; they are not fixed counts.
