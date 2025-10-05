# Configuration Reference

This reference lists all configuration sections, keys, valid values, and how to set them. Use YAML files under `configs/` and pass them via `--config`.

How to use
- Place a YAML override in `configs/`, e.g. `configs/classification.yaml`.
- Run with: `deepfake-cli --train --task classification --config configs/classification.yaml`
- Defaults are defined in `src/deepfake/config/default.yaml`. Any keys you omit fall back to these defaults.

Data
- `dataset_name` (str): Hugging Face dataset ID. Example: `saberzl/SID_Set`.
- `image_size` (int): Square side length in pixels for preprocessing.
- `train_samples` (int): Number of training samples to use.
- `val_samples` (int): Validation samples to use.
- `test_samples` (int): Test samples to use.
- `use_streaming` (bool): Use HF streaming mode for large datasets.
- `use_disk_cache` (bool): Cache derived splits to disk for reuse.

Loader
- `batch_size` (int): Batch size for DataLoaders.
- `shuffle_train` (bool): Shuffle training set.
- `shuffle_val` (bool): Shuffle validation set.
- `shuffle_test` (bool): Shuffle test set.
- `num_workers` (int): PyTorch DataLoader workers.

Model
- `class_names` (list[str]): Ordered list of classes for classification.
- `num_classes` (int): Derived automatically from `class_names` (do not set).
- `normalize_mean` (list[float]): Per‑channel mean for normalization.
- `normalize_std` (list[float]): Per‑channel std for normalization.
- `tampered_label` (int): Label index considered “tampered” for segmentation splits.

Training
- `epochs` (int): Number of training epochs.
- `learning_rate` (float): Default learning rate; used if not overridden in optimizer params.
- `seed` (int): Random seed for Python, NumPy, PyTorch.
- `device` (str): `cuda` or `cpu`. If omitted, auto‑detected at runtime.
- `save_interval` (int): Save a checkpoint every N epochs.
- `optimizer` (object): Optimizer configuration
  - `name` (str): One of: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `adadelta`, `adamax`, `asgd`, `rprop`, `lbfgs`.
    - If available in your PyTorch version: `radam`, `nadam`, `sparseadam`.
    - Name is case‑insensitive; dashes/underscores are ignored (e.g., `AdamW`, `adam_w`).
  - `params` (dict): Hyperparameters forwarded to the selected optimizer. Do not include `params` here — it’s provided by the model.
    - Common examples:
      - SGD: `momentum`, `nesterov`, `weight_decay`
      - Adam/AdamW: `betas`, `eps`, `weight_decay`
      - RMSprop: `alpha`, `weight_decay`, `momentum`
      - LBFGS: `lr`, `max_iter`, `history_size`
    - If `lr` is omitted, `training.learning_rate` is used.

Paths
- `model_path` (str): Destination for the “best model” checkpoint.
- `results_dir` (str): Directory for plots and JSON metrics.
- `history_file` (str): Filename for metrics JSON inside `results_dir`.
- `logging_dir` (str|null): Directory for logs. Set to `null` to disable.
- `run_name` (str|null): Optional subdirectory to group outputs per run. If omitted, a timestamp is generated. You can also set env `RUN_NAME` to force a name.

Setting values
- YAML override file example:
  - training:
      epochs: 20
      device: cuda
      optimizer:
        name: AdamW
        params:
          lr: 5e-4
          weight_decay: 0.01
- Inspect the merged configuration for a given override:
  - `python -m deepfake.utils.config_loader --config configs/classification.yaml`
