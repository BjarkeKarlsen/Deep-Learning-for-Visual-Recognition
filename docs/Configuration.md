# Configuration Reference

This page lists every configuration block the toolkit understands, along with the keys you can set in your YAML files. Each section shows the expected value type and the default delivered in `src/deepfake/config/default.yaml`.

## Using A Config File
- Copy one of the samples in `configs/` and adjust the keys you need.
- Run the CLI with the file: `deepfake-cli train --task classification --config configs/my-run.yaml`.
- Any key you omit falls back to the defaults below.

---

## `data`
Controls dataset selection and sampling.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `saberzl/SID_Set` | Hugging Face dataset identifier. |
| `image_size` | int | `224` | Square edge length after transforms. |
| `train_samples` | int | `10` | Max samples pulled into the training split (omit for full dataset when not streaming). |
| `val_samples` | int | `10` | Max validation samples. |
| `test_samples` | int | `10` | Max test samples. |
| `use_streaming` | bool | `true` | Stream data from HF (requires explicit sample caps). |
| `use_disk_cache` | bool | `true` | Cache derived splits on disk for reuse. |

---

## `loader`
PyTorch `DataLoader` knobs.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `batch_size` | int | `4` | Mini-batch size for all splits. |
| `shuffle_train` | bool | `true` | Shuffle training batches. |
| `shuffle_val` | bool | `false` | Shuffle validation sampler. |
| `shuffle_test` | bool | `false` | Shuffle test sampler. |
| `num_workers` | int | `4` | Worker processes per loader. |

---

## `model`
Pre-processing and label handling.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `class_names` | list[str] | `["Real","Synthetic","Tampered"]` | Ordered class labels (classification task). |
| `normalize_mean` | list[float] | `[0.485,0.456,0.406]` | Channel-wise normalisation mean. |
| `normalize_std` | list[float] | `[0.229,0.224,0.225]` | Channel-wise normalisation std. |
| `tampered_label` | int | `2` | Index used as the tampered class (segmentation). |

`num_classes` is computed automatically from `class_names`.

---

## `training`
Hyperparameters shared across tasks.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `epochs` | int | `10` | Number of training epochs. |
| `learning_rate` | float | `0.001` | Base learning rate passed to the optimizer. |
| `seed` | int | `42` | Global RNG seed. |
| `device` | str | Auto-set (`"cuda"` if available, else `"cpu"`). Override only when needed. |
| `save_interval` | int | `5` | Save a checkpoint every N epochs (classification). |
| `optimizer` | mapping | see below | Configure the optimiser family and hyperparameters. |

### `training.optimizer`
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | str | `adam` | Supported values: `adam`, `adamw`, `sgd`. Comparison is case-insensitive. |
| `weight_decay` | float | `0.0` | L2 weight decay (applies to all optimisers). |
| `betas` | list[float] | `[0.9, 0.999]` | Only used for Adam/AdamW; must contain two floats. |
| `momentum` | float | `0.9` | Used when `name: sgd`. |
| `nesterov` | bool | `false` | Enable Nesterov momentum for SGD. |

Example override:

```yaml
training:
  learning_rate: 0.0003
  optimizer:
    name: adamw
    weight_decay: 0.01
    betas: [0.9, 0.95]
```

---

## `paths`
Output locations. Paths are expanded relative to the repository root.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `base` | str | `outputs` | Root directory for run artefacts. |
| `task` | str | `""` | Filled in automatically by the CLI (`classification`/`segmentation`). |
| `model_filename` | str | `best_model.pth` | Filename used when saving checkpoints. |
| `history_filename` | str | `history.json` | Training history file. |
| `metrics_filename` | str | `evaluation_metrics.json` | Evaluation results file. |
| `log_filename` | str | Timestamped by default | Logfile name within the run directory. |
| `args_filename` | str | `args.json` | Stores CLI arguments for auditing. |

---

## Checklist When Editing Configs
1. Duplicate a template under `configs/` so you keep a record for each run.
2. Adjust the sections above—most experiments change `data`, `training`, and `training.optimizer`.
3. Point the CLI to your file with `--config`.
4. Keep output paths unique per run so results don’t overwrite each other.

That’s it: the dataclasses enforce types, defaults fill in the gaps, and any YAML override you supply becomes the source of truth for a run.
