# Deepfake Detection Toolkit

Two lightweight PyTorch pipelines for the Hugging Face [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) dataset:

- **Classification** – detects `Real`, `Synthetic`, or `Tampered` frames via a simple CNN.
- **Segmentation** – highlights the tampered regions using a compact U-Net.

Both share the same configuration system, logging utilities, and Hugging Face data manager.

---

## 1. Environment Setup

```bash
python -m venv .venv           # or use conda
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The code auto-selects `cuda` when a GPU is available; otherwise it falls back to `cpu`.

---

## 2. Configuration Basics

Configuration is driven by `src/config/default.yaml` plus an explicit override file that you pass via `--config`.

Key sections:
- `data`: image size, maximum sample counts (`null` to use the full split), streaming and caching toggles, dataset name.
- `loader`: batch size, shuffle flags, number of workers.
- `model`: class names, normalisation stats, tampered class label (used by segmentation filtering).
- `training`: epochs, learning-rate, random seed (respected by `main.py`).
- `paths`: output locations for checkpoints, results, and optional logging directory.

Helpful utilities:
- Preview the effective config: `python -m src.utils.config_loader --config classification.yaml`
- Dump the default template: `python -m src.utils.config_loader --dump-default --output my-config.yaml`

> **Streaming note:** the default config keeps `data.use_streaming: true` to avoid downloading the full dataset. The pipelines need random-access datasets; when you set `train/val/test_samples: null` with streaming enabled, the loader materialises the consumed split in memory. Switch `use_streaming` to `false` if you plan to run on the full dataset and want Hugging Face to manage an on-disk cache instead.

---

## 3. Running the Pipelines

### 3.1 Classification

```bash
# train (writes metrics + best checkpoint)
python main.py --train --task classification --config classification.yaml

# evaluate using the latest trained weights
python main.py --eval --task classification --config classification.yaml
```

Each batch yielded by the classifier dataset is a dictionary with `"image"`, `"label"`, and a dummy `"mask"` tensor (mask is present only for api consistency).

### 3.2 Segmentation

```bash
# train the U-Net on tampered examples only
python main.py --train --task segmentation --config segmentation.yaml

# evaluate Dice / IoU on the tampered test split
python main.py --eval --task segmentation --config segmentation.yaml
```

Segmentation splits automatically filter to samples whose `label` matches `model.tampered_label` **and** contain a mask.

### 3.3 Custom Config

Pick any YAML file that matches the schema, then pass it explicitly. For example:

```bash
python main.py --train --task classification --config configs/large-run.yaml
```

---

## 4. Outputs and Logging

- **Checkpoints** – `cfg.paths.model_path` (default `models/best_model.pth`).
- **Metrics & history** – JSON files and plots under `cfg.paths.results_dir`.
- **Logs** – timestamped files in `logs/` or `cfg.paths.logging_dir` when provided.

Segmentation and classification both persist Matplotlib figures (confusion matrix, metric bar charts, training curves) alongside their metric JSON payloads.

---

## 5. Troubleshooting & Tips

- Set `training.seed` in your config to reproduce runs. `main.py` honours the value at startup.
- When using streaming with large sample caps, consider enabling `data.use_disk_cache` to avoid repeatedly materialising splits.
- If Hugging Face credentials are required (private datasets), log in via `huggingface-cli login` before running the scripts.

Further reference is available in `docs/Configuration.md`, `docs/DataManager.md`, and `docs/Dataset.md`.
