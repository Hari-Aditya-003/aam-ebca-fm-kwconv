# Results

This directory publishes lightweight, aggregate experiment evidence only. The source video, extracted frames, augmentations, labels, dataset previews, model checkpoints, and generated videos are intentionally excluded from Git.

## Training snapshot

Snapshot time: **30 September 2026, 20:56 IST** (`2026-09-30T15:26:45Z`). Training was active and had completed **65/300 epochs (21.7%)**.

| Metric | Latest value (epoch 65) | Best value observed | Epoch of best |
|---|---:|---:|---:|
| Precision | 0.80084 | 0.82107 | 36 |
| Recall | 0.71182 | 0.71746 | 48 |
| mAP@0.50 | 0.79950 | 0.81440 | 28 |
| mAP@0.50:0.95 | 0.57659 | 0.58744 | 24 |

The best checkpoint at this snapshot corresponds to epoch 24 under the Ultralytics detection fitness score. Training is unfinished, so these values may change.

## Published artifacts

- [`status/live_training_status.json`](status/live_training_status.json): portable status and metric summary.
- [`training/aam_ebca_fm_kwconv_green/results.csv`](training/aam_ebca_fm_kwconv_green/results.csv): one row per completed epoch through epoch 65.
- [`architecture/module_smoke_test.json`](architecture/module_smoke_test.json): output shapes and AAM → EBCA → FM → KWConv module order.
- [`hardware/runtime.json`](hardware/runtime.json): recorded CUDA, PyTorch, and GPU runtime.

Final validation curves, confusion matrices, inference tables, and tracked-video summaries will be added only after the configured 300-epoch run completes. Large `.pt` checkpoints and videos remain local.

## Interpretation

The validation labels were generated from COCO-pretrained detector proposals. These metrics measure agreement against pseudo-labels, not performance against independently annotated ground truth. A defensible final precision, recall, F1, or mAP claim requires a separate human-annotated evaluation set.
