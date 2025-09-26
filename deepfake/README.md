# Deepfake Detection Toolkit

Two lightweight PyTorch pipelines built on the Hugging Face [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) dataset:

- **Classification** – predicts `Real`, `Synthetic`, or `Tampered` frames with a compact CNN.
- **Segmentation** – localises tampered regions with a slim U-Net.

Both pipelines share the same YAML-driven configuration system, logging utilities, and Hugging Face data manager.

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

## 2. Configuration (YAML Required)

- Every run must point to a `.yaml` config with `--config <file>`. JSON/TOML/etc. are not supported.
- Default configs live at the project root:
  - `classification.yaml` – tailored to `--task classification`.
  - `segmentation.yaml` – tailored to `--task segmentation`.
- Copy either file if you need a variant, e.g. `cp classification.yaml configs/large-batch.yaml`, then update paths and hyperparameters inside the YAML.
- Keep `paths.model_path`, `paths.results_dir`, and `paths.logging_dir` unique per pipeline so checkpoints and logs never collide. The provided defaults already isolate outputs under `models/<task>/` and `results/<task>/`.
- Streaming is enabled by default via `data.use_streaming: true` to avoid downloading the entire dataset. Switch it to `false` when you want Hugging Face to manage an on-disk cache instead.

See `docs/Configuration.md` for field-by-field details.

---

## 3. Run Pipelines

### Classification

```bash
# train (writes metrics + best checkpoint)
python main.py --train --task classification --config classification.yaml

# evaluate with the latest trained weights
python main.py --eval --task classification --config classification.yaml
```

Each batch contains `"image"`, `"label"`, and a placeholder `"mask"` tensor for API consistency.

### Segmentation

```bash
# train the U-Net on tampered examples only
python main.py --train --task segmentation --config segmentation.yaml

# evaluate Dice / IoU on the tampered test split
python main.py --eval --task segmentation --config segmentation.yaml
```

Segmentation splits automatically filter to samples whose `label` matches `model.tampered_label` and that include a mask.

---

## 4. Outputs and Logging

- **Checkpoints** – saved under `cfg.paths.model_path` (default `models/<task>/best_model.pth`).
- **Metrics & plots** – written to `cfg.paths.results_dir`, including Matplotlib curves and confusion matrices.
- **Logs** – timestamped files in `logs/` or the override specified in `cfg.paths.logging_dir`.

---

## 5. Troubleshooting & Tips

- Set `training.seed` in your YAML to reproduce runs; `main.py` applies it at startup.
- For large streaming jobs, enable `data.use_disk_cache` to avoid re-materialising splits in memory.
- Authenticate with `huggingface-cli login` if the dataset or models require credentials.
