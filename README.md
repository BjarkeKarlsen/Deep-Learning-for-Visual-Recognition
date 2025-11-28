# Deepfake Detection Toolkit

Two lightweight PyTorch pipelines built on the Hugging Face [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) dataset:

- **Classification** – predicts `Real`, `Synthetic`, or `Tampered` frames with a compact CNN.
- **Classification (ResNet option)** – set `model.backbone.name` to `resnet18`, `resnet34`, or `resnet50` to swap in a pretrained backbone while keeping the lightweight training loop.
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
- `data.use_streaming` switches the loaders into Hugging Face streaming mode. Keep the per-split sample caps finite—the manager buffers the requested amount into memory before handing it to PyTorch.
- `data.use_disk_cache` is currently a no-op and kept only for backwards compatibility with older configs.
- `data.augment` enables optional training-time augmentations (e.g., random crops/flips/jitter) when `enable: true`; set `preview_samples` to dump an augmentation preview image in the run folder.
- `training.label_smoothing`, `training.grad_clip_norm`, and `training.ema_decay` tighten optimisation; use `training.scheduler.name: "cosine"` (or `onecycle`) to enable LR schedules.

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

## 4. Hyperparameter Tuning

We ship a small Optuna harness that reuses the existing training pipeline to search for better hyperparameters.

Run the tuner (examples below target the lightweight `dev` configs for quick iterations):

   **Classification**
   ```bash
   deepfake-cli tune --task classification --env dev --n-trials 30 --device cuda
   ```

   **Segmentation**
   ```bash
   deepfake-cli tune --task segmentation --env dev --n-trials 30 --device cuda
   ```

Key flags:
   - `--n-trials` / `--timeout` control the budget.
   - `--storage` and `--study-name` let you resume studies (defaults to `sqlite:///optuna_study.db`).
   - `--pruner` enables Optuna’s median pruner to stop weak trials early.
   - `--keep-runs` preserves the generated `outputs/<task>/runs/<timestamp>_trial#` folders for manual inspection (otherwise they are cleaned automatically once the metric is recorded).

Inspect results:
   ```bash
   optuna-dashboard sqlite:///optuna_study.db
   ```

The tuner samples learning-rate schedule parameters, optimiser regularisation, and augmentation probabilities (classification) / loss settings (segmentation). It writes the best validation metric back to the study and prints the top-performing hyperparameters when complete.

---

## 5. Troubleshooting & Tips

- Set `training.seed` in your YAML to reproduce runs; the CLI applies it at startup.
- When streaming, ensure `data.train_samples`, `data.val_samples`, and `data.test_samples` are set to realistic limits so the loader can materialise the stream.
- Authenticate with `huggingface-cli login` if the dataset requires credentials.
- Use `python -m deepfake.utils.config_loader --config <path>` to print the merged configuration for debugging.
