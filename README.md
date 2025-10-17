# Deepfake Detection Toolkit

Two lightweight PyTorch pipelines built on the Hugging Face [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) dataset:

- **Classification** – predicts `Real`, `Synthetic`, or `Tampered` frames with a compact CNN.
- **Segmentation** – localises tampered regions with a slim U-Net.

Both share a common configuration system, logging utilities, dataset manager, and output conventions.

---

## Repository Layout

- `src/deepfake/` – package containing the task pipelines, data helpers, visualisation utilities, and configuration schema.
- `configs/` – runnable YAML overrides (e.g. `classification.yaml`, `segmentation.yaml`).
- `outputs/` – default destination for logs, checkpoints, and result artefacts (safe to wipe between runs).
- `docs/` – API and configuration documentation.
- `notebooks/` – exploratory Jupyter notebooks.

---

## 1. Environment Setup
This guide walks you through setting up the environment for using the Deepfake Toolkit, including both CPU and GPU configurations.

### 1. Conda
Create a new Conda environment with Python 3.12:
```bash
conda create -n deepfake-env python=3.12 -y
```

Activate the environment:
```bash
conda activate deepfake-env
```

#### 2. Editable Installation (Development Mode)
If you want to install the package in editable mode for development purposes, use one of the following commands depending on your hardware:

```bash
pip install -e .
```
Or
```bash
conda install pytorch torchvision pytorch-cuda=12.8 -c conda-forge -c nvidia
pip install -e .
```
#### 3. Using a Virtual Environment (Optional)

If you prefer using a Python virtual environment instead of Conda:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 7. Notes

- CUDA support is auto-detected when available; otherwise, the toolkit defaults to CPU.
- Choose the installation method (CPU or GPU) based on your hardware capabilities.
- For Conda users, the recommended Python version is 3.12 to ensure compatibility.

---

## 2. Configuration

- Configs live under `configs/`. The CLI picks the file based on `--task` and `--env`:
  - `--task classification --env dev` → `configs/dev-classification.yaml`
  - `--task segmentation --env test` → `configs/segmentation.yaml`
- `src/deepfake/config/default.yaml` holds the baseline values. Copy a config from `configs/` if you need a custom variant (e.g. `cp configs/classification.yaml configs/classification-large.yaml`) and edit the copy.
- Paths are derived inside the config via `paths.*`; no need to supply extra CLI flags. Each run gets a timestamped root (`paths.run_id`) under `outputs/<task>/runs/`.
- `data.use_streaming` is supported but may fall back to local caching depending on the Hugging Face dataset backend.
- `data.augment` enables optional training-time augmentations (e.g., random crops/flips/jitter) when `enable: true`; set `preview_samples` to dump an `augmentation_preview.png` in the run folder.

See `docs/Configuration.md` for field-by-field details.

---

## 3. Run Pipelines
Once you’ve installed the package (e.g. via `pip install -e .`), the `deepfake-cli` entry point is available on your PATH. Below are the primary commands and options:

Train a model

```bash
deepfake-cli train --task <classification|segmentation> --env <dev|test> [--runid <RUN_ID>] [--checkpoint <epoch>]
```

Example (fresh classification run in dev):

```bash
deepfake-cli train --task classification --env dev
```

Example (segmentation in test):

```bash
deepfake-cli train --task segmentation --env test
```

Evaluate a trained model

```bash
deepfake-cli eval --task <classification|segmentation> --runid <RUN_ID> [--env <dev|test>]
```

Example (eval segmentation run):

```bash
deepfake-cli eval --task segmentation --runid 20251010T130000Z --env test
```

If you omit `--runid`, the CLI now inspects `outputs/<task>/runs/` and evaluates the most recent run that contains a saved model (plotting uses the latest run with training/eval history).

Plot metrics from history or evaluation

```bash
deepfake-cli plot --task <classification|segmentation> --stage <train|eval> [--runid <RUN_ID>] [--env <dev|test>]
```

Plot training curves for classification (dev):

```bash
deepfake-cli plot --task classification --stage train --env dev
```

Plot evaluation confusion matrix for classification (test):

```bash
deepfake-cli plot --task classification --stage eval --runid 20251010T130000Z --env test
```

Plot training history for segmentation (dev):

```bash
deepfake-cli plot --task segmentation --stage train --env dev
```
### 3.1 Options

`--task` selects the pipeline.
`--env` chooses which YAML config (`dev-classification.yaml` or `test-segmentation.yaml`) to load.
`--runid` lets you point to a previous run’s folder under `outputs/<task>/runs/<run_id>`.
`--stage` (plot only) picks training vs. evaluation metrics.

### 3.2 Resuming a Run

To continue a partially finished training run (e.g. stopped at epoch 31/50):

1. Locate the run folder, e.g. `outputs/classification/runs/20250314T102200Z/`.
2. Find the latest checkpoint inside `checkpoints/` (e.g. `epoch_030`).
3. Launch training with both the run-id and checkpoint:

```bash
deepfake-cli train --task classification --env dev \
  --runid 20250314T102200Z \
  --checkpoint 30
```

The trainer restores model weights, optimizer state, and history, then resumes at epoch 31. Omit `--checkpoint` to start over within the same run directory.

---

## 4. Troubleshooting & Tips

- Set `training.seed` in your YAML to reproduce runs; the CLI applies it at startup.
- For large streaming jobs, keep `data.use_disk_cache: true` to avoid re-materialising derived splits.
- Authenticate with `huggingface-cli login` if the dataset requires credentials.
- Use `python -m deepfake.utils.config_loader --config <path>` to print the merged configuration for debugging.
