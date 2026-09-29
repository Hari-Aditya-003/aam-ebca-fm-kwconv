# AAM, EBCA, FM, KWConv

Green/yellow apple detection, relative-depth localization, and duplicate-safe orchard tracking.

## Research objectives

- Detect small, overlapping, and partially occluded apples with AAM, EBCA, FM, and KWConv feature enhancement.
- Use AdaBins-derived relative depth to separate foreground from background fruit and support association.
- Maintain IDs with a two-stage, depth-aware ByteTrack-style tracker and count each valid track only once.

## Run the notebook

Open [AAM, EBCA, FM, KWConv 100.ipynb](<Codeing/AAM, EBCA, FM, KWConv 100.ipynb>) and run its cells in order. The reusable implementation is [fruit_pipeline.py](Codeing/fruit_pipeline.py).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r Codeing/requirements.txt
.venv/bin/python -m ipykernel install --user --name fruitvision --display-name "FruitVision (.venv)"
```

The main environment includes the CPU builds of PyTorch, Ultralytics YOLOv8, JupyterLab, LabelImg, OpenCV, and SciPy. The local install also includes the official outdoor AdaBins source/checkpoint in `third_party/AdaBins/`; it is excluded from Git because the checkpoint is large and AdaBins is GPL-3.0. Use its output only as relative depth until the orchard camera is calibrated.

The workflow saves extracted frames, seed candidates, manually verified labels, augmentations, and YOLO splits under `Dataset/`; it saves architecture checks, training outputs, metrics, annotated videos, track CSVs, and summaries under `Results/`.

`DJI_0418.MP4` currently cannot be decoded because its MP4 index (`moov` atom) is missing. Re-export it from the source device as a playable H.264 MP4 and replace `Dataset/DJI_0418.MP4` before running extraction. The notebook detects this safely and does not overwrite the source video.

## Data policy

`Dataset/` and `Results/` are local-only and excluded from Git. Green/yellow HSV candidates are only a review aid—green foliage can be confused with green apples—so train only on labels saved in `Dataset/labels/verified/`. Splitting occurs before training augmentation to prevent temporal leakage from neighbouring video frames.
