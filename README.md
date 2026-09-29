# AAM, EBCA, FM, KWConv

Green/yellow apple detection, relative-depth localization, and duplicate-safe orchard tracking.

## Research objectives

- Detect small, overlapping, and partially occluded apples with AAM, EBCA, FM, and KWConv feature enhancement.
- Use AdaBins-derived relative depth to separate foreground from background fruit and support association.
- Maintain IDs with a two-stage, depth-aware ByteTrack-style tracker and count each valid track only once.

## Run the notebook

Open [AAM, EBCA, FM, KWConv 100.ipynb](<Codeing/AAM, EBCA, FM, KWConv 100.ipynb>) and run its cells in order. The reusable implementation is [fruit_pipeline.py](Codeing/fruit_pipeline.py).

```bash
python -m pip install -r Codeing/requirements.txt
```

The workflow saves extracted frames, seed candidates, manually verified labels, augmentations, and YOLO splits under `Dataset/`; it saves architecture checks, training outputs, metrics, annotated videos, track CSVs, and summaries under `Results/`.

`DJI_0418.MP4` currently cannot be decoded because its MP4 index (`moov` atom) is missing. Re-export it from the source device as a playable H.264 MP4 and replace `Dataset/DJI_0418.MP4` before running extraction. The notebook detects this safely and does not overwrite the source video.

## Data policy

`Dataset/` and `Results/` are local-only and excluded from Git. Green/yellow HSV candidates are only a review aid—green foliage can be confused with green apples—so train only on labels saved in `Dataset/labels/verified/`. Splitting occurs before training augmentation to prevent temporal leakage from neighbouring video frames.
