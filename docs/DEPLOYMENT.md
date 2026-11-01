# Apple Detection: reproducible setup and operation

This is a notebook-led research pipeline. Deployment means preparing a workstation or GPU machine to reproduce training, validation, and video exports. The repository does not provide a hosted prediction API or a browser application.

## 1. Retrieve source and artifacts

Install Git LFS, then run:

```bash
git lfs install
git clone https://github.com/Hari-Aditya-003/aam-ebca-fm-kwconv.git
cd aam-ebca-fm-kwconv
git lfs pull
git lfs ls-files
git rev-parse HEAD
```

Keep the commit SHA with each experiment. Checkpoints, prediction tables, and final videos use LFS; downloading a ZIP may leave pointer files instead of usable artifacts. If a file begins with `version https://git-lfs.github.com/spec/v1`, retrieve its LFS object before using it.

## 2. Prepare Python

Use a Python version compatible with the selected PyTorch and Ultralytics releases.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Codeing/requirements.txt
fruitvision_kernel="fruitvision-$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
python -m ipykernel install --user --name "$fruitvision_kernel" --display-name "FruitVision ($fruitvision_kernel)"
```

For full training, install the CUDA-enabled PyTorch build appropriate to the machine's NVIDIA driver. Run the notebook's runtime diagnostic cell and confirm CUDA works before starting an expensive run. The requirement ranges are not a frozen environment; record the resolved versions:

```bash
experiment_env="$(mktemp experiment-environment.XXXXXX)"
python -m pip freeze > "$experiment_env"
```

Keep this file with the experiment's results. Installing the requirements alone does not supply the video, dataset, or AdaBins checkpoint.

## 3. Supply local inputs

| Input | Expected location | Availability |
| --- | --- | --- |
| Source orchard video | `Dataset/DJI_0418.MP4` | Must be supplied locally |
| Generated dataset | `Dataset/` | Created by the notebook; excluded from Git |
| Detector checkpoints | `Results/training/aam_ebca_fm_kwconv_green/weights/` | Git LFS artifacts |
| AdaBins source and model | `third_party/AdaBins/pretrained/AdaBins_kitti.pt` | Separate optional setup |

Allow space for the generated dataset, approximately 36 GB according to the recorded run, plus exports and checkpoints. AdaBins depth is relative/estimated; it is not calibrated distance in metres.

## Optional AdaBins setup

Clone the [official AdaBins source](https://github.com/shariqfarooq123/AdaBins) into the expected location:

```bash
mkdir -p third_party
git clone https://github.com/shariqfarooq123/AdaBins.git third_party/AdaBins
python -m pip install matplotlib
```

Install a `torchvision` build matching your selected PyTorch/CUDA build. The adapter also uses the NumPy, SciPy, Pillow, and tqdm packages from the project requirements. Follow upstream model-loading requirements; the EfficientNet backbone may require a first-run network download.

Download `AdaBins_kitti.pt` using the pretrained-model link in the official AdaBins README and place it at `third_party/AdaBins/pretrained/AdaBins_kitti.pt`. The adapter changes into the AdaBins source directory before creating `InferenceHelper(dataset="kitti")`, so the upstream relative checkpoint path resolves there. Record the upstream source SHA and checkpoint provenance with the experiment.

If AdaBins is not installed, keep `RUN_ADABINS = False` in the main notebook and set `USE_ADABINS_RELATIVE_DEPTH = False` in the Results and Status notebook before enabling `RUN_VIDEO_EXPORTS`. That export notebook defaults depth to enabled and will otherwise fail on the missing checkpoint.

## 4. Execute the workflow deliberately

Start Jupyter from the repository root:

```bash
python -m jupyterlab
```

Open [the main notebook](../Codeing/AAM,%20EBCA,%20FM,%20KWConv%20100.ipynb), choose the uniquely named FruitVision kernel created above, and confirm the root diagnostic points at this checkout. If the notebook kernel starts inside `Codeing/`, set its working directory to the repository root before the path-setup cell; the exported source otherwise falls back to the original author's workstation path.

Run cells in order. Review each `RUN_...` and `REBUILD_...` flag before enabling a stage. The workflow creates frames, augmentations, grouped splits, pseudo-labels, model modules, training results, validation, and optional depth/tracking outputs. Use a separate checkout or archive previous results before regeneration.

For inspecting completed results without retraining, use [Results and Status](../Codeing/AAM,%20EBCA,%20FM,%20KWConv%20100%20-%20Results%20and%20Status.ipynb) and [Results/README.md](../Results/README.md).

## 5. Rebuild annotated videos

With the source video, LFS CSVs, Python dependencies, and FFmpeg installed:

```bash
python Codeing/render_result_videos.py --help
python Codeing/render_result_videos.py --output-dir Results/inference-local
```

The dedicated output directory preserves published videos. The default encoder is `libx264`; `h264_nvenc` requires working NVIDIA video-encoding support.

## Verification before sharing

- Record source commit, environment, input-video identity, random settings, GPU, and enabled flags.
- Confirm original images and their augmentations stay in the same split.
- Distinguish validation against pseudo-labels from independently annotated evaluation.
- Verify checkpoint loading with the custom model definitions registered in the workflow.
- Open exported videos and check row counts, frame alignment, and tracker IDs.
- State explicitly whether depth is enabled and whether independent ground truth exists.

The published 300-epoch run is historical evidence, not a guarantee that another environment reproduces identical outputs. See the [technical documentation](../Documentation/AAM_EBCA_FM_KWConv_Technical_Documentation.md).

## Recovery

If a stage fails, preserve its logs and partial outputs, correct the prerequisite, and rerun only the necessary stages. Do not enable rebuild flags blindly. Keep an earlier checkpoint and result set in a separate directory so recovery does not overwrite the baseline.
