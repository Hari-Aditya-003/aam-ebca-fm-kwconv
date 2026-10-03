# Results

This directory publishes the completed aggregate results plus final videos and detector checkpoints through Git LFS. Source frames, augmentations, labels, dataset previews, and third-party AdaBins files remain excluded.

## Completed training and validation

The enhanced detector completed **300/300 epochs**. The selected checkpoint is `best.pt` from epoch 24.

| Metric | Final `best.pt` validation |
|---|---:|
| Precision | 0.80431 |
| Recall | 0.71415 |
| F1 | 0.75649 |
| mAP@0.50 | 0.80977 |
| mAP@0.50:0.95 | 0.58741 |

## Published artifacts

- [`status/live_training_status.json`](status/live_training_status.json): portable status and metric summary.
- [`training/aam_ebca_fm_kwconv_green/results.csv`](training/aam_ebca_fm_kwconv_green/results.csv): one row for each of the 300 completed epochs.
- [`architecture/module_smoke_test.json`](architecture/module_smoke_test.json): output shapes and AAM → EBCA → FM → KWConv module order.
- [`hardware/runtime.json`](hardware/runtime.json): recorded CUDA, PyTorch, and GPU runtime.
- [`csv/final_validation_metrics.csv`](csv/final_validation_metrics.csv): final best-checkpoint validation metrics.
- [`graphs/`](graphs/): F1, precision, recall, PR, and confusion-matrix plots.
- [`inference/apples_id_depth.mp4`](inference/apples_id_depth.mp4), [`apples_confidence_depth.mp4`](inference/apples_confidence_depth.mp4), and [`apples_id_confidence.mp4`](inference/apples_id_confidence.mp4): final annotated videos.
- [`csv/apple_detections_confidence_depth.csv`](csv/apple_detections_confidence_depth.csv) and [`apple_tracks_id_confidence_depth.csv`](csv/apple_tracks_id_confidence_depth.csv): frame-level corrected relative-depth and tracking tables.
- `training/aam_ebca_fm_kwconv_green/weights/{best,last}.pt`: LFS-hosted model checkpoints.

The displayed `D` value is relative depth, not metres. It combines high-resolution AdaBins output with a conservative apparent-apple-scale correction because raw AdaBins smoothed small fruit into surrounding foliage. Recreate the videos from the published tables with `python Codeing/render_result_videos.py`; use `--encoder h264_nvenc` when NVIDIA FFmpeg encoding is available.

## Interpretation

The validation labels were generated from COCO-pretrained detector proposals. These metrics measure agreement against pseudo-labels, not performance against independently annotated ground truth. A defensible final precision, recall, F1, or mAP claim requires a separate human-annotated evaluation set.
