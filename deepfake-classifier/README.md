# Deepfake Detection CNN
Simple 3-class classifier (Real / Synthetic / Tampered) built around the Hugging Face `saberzl/SID_Set` dataset.

## Quick start
```bash
python -m venv .venv            # or use conda
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# train the CNN (saves weights + metrics)
python main.py --train

# evaluate with the saved weights
python main.py --eval
```
The script auto-selects GPU (`cuda`) if available; otherwise it stays on CPU.

## How data is handled
- Reads `saberzl/SID_Set` via Hugging Face datasets.
- Streams by default (`data.use_streaming: true`) so nothing is downloaded. Flip to `false` to keep a local cache (`~/.cache/huggingface/datasets/` by default).
- Optional disk caching (`data.use_disk_cache: true`) saves any custom splits under `~/.cache/huggingface/datasets/custom_splits/SID/`.

## Using `config.yaml`
- `src/config/default.yaml` is the template; `config.yaml` overrides it. Use `--config <file>` to point at another file.
- Most-used fields:
  - `data`: sample counts (`null`=full split), `image_size`, streaming + caching toggles.
  - `loader`: batch size, shuffles, worker count.
  - `training`: epochs, learning rate, seed (device auto-fills).
  - `paths`: output locations for weights, results, logs.
- Preview effective settings: `python -m src.utils.config_loader [--config path]`.
- Need a fresh copy? `python -m src.utils.config_loader --dump-default --output my-config.yaml` (`--force` to overwrite).

## What gets produced
- Best checkpoint: `models/best_model.pth` (or your `paths.model_path`).
- Metrics & plots: JSON + PNG files in `results/` by default.
- Logs: timestamped files in `logs/` unless `paths.logging_dir` says otherwise.

More detail sits in `docs/Configuration.md`, `docs/DataManager.md`, and `docs/Dataset.md`.
