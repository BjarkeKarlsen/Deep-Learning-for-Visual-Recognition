# Configuration Reference

This page lists every configuration block the toolkit understands, along with the keys you can set in your YAML files. Each section shows the expected value type and the default delivered in `src/deepfake/config/default.yaml`.

## Using Configuration Files
- Copy one of the samples in `configs/` and adjust the keys you need.
- The CLI selects a config automatically from `configs/` based on `--task` and `--env`. For example `deepfake-cli train --task classification --env dev` loads `configs/dev-classification.yaml`.
- If you create a custom override, point to it with `--config` (optional) or update the `--env`/file naming convention accordingly.
- Any key you omit falls back to the defaults below.

---

## `data`
Controls dataset selection and sampling.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `saberzl/SID_Set` | Hugging Face dataset identifier. |
| `image_size` | int | `224` | Square edge length after transforms. |
| `train_samples` | int | `10` | Max samples pulled into the training split (must stay finite when streaming). |
| `val_samples` | int | `10` | Max validation samples. |
| `test_samples` | int | `10` | Max test samples. |
| `use_streaming` | bool | `true` | Stream data from HF; you must keep the sample caps above finite because the loader buffers the stream into memory. |
| `use_disk_cache` | bool | `true` | Reserved for future use. The current implementation ignores this flag. |
| `augment.enable` | bool | `false` | Toggle training-time augmentation for both pipelines. |
| `augment.random_resized_crop` | bool | `true` | Use random resized crop instead of a plain resize when augmentations are enabled. |
| `augment.scale_min` / `scale_max` | float | `0.8` / `1.0` | Lower/upper bounds for crop area scale. |
| `augment.horizontal_flip_prob` | float | `0.5` | Probability of horizontal flips. |
| `augment.color_jitter_*` | float | see default YAML | Brightness/contrast/saturation/hue jitter magnitudes. |
| `augment.gaussian_blur_prob` | float | `0.0` | Probability of applying Gaussian blur. |
| `augment.random_erasing_*` | float | see default YAML | Parameters for random erasing; ignored if probability is zero. |
| `augment.preview_samples` | int | `0` | Number of augmented samples to render into a preview grid when augmentations are enabled. |
| `augment.preview_seed` | int | `1234` | RNG seed for augmentation preview sampling. |

---

## `loader`
PyTorch `DataLoader` knobs.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `batch_size` | int | `4` | Mini-batch size for all splits. |
| `shuffle_train` | bool | `true` | Shuffle training batches. |
| `shuffle_val` | bool | `false` | Shuffle validation sampler. |
| `shuffle_test` | bool | `false` | Shuffle test sampler. |
| `num_workers` | int | `4` | Worker processes per loader (set to `0` automatically when streaming). |
| `prefetch_factor` | int | `2` | Passed to PyTorch `DataLoader` when `num_workers > 0`. |
| `persistent_workers` | bool | `false` | Keep worker processes alive between epochs. |

---

## `model`
Pre-processing and label handling.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `class_names` | list[str] | `["Real","Synthetic","Tampered"]` | Ordered class labels (classification task). |
| `normalize_mean` | list[float] | `[0.485,0.456,0.406]` | Channel-wise normalisation mean. |
| `normalize_std` | list[float] | `[0.229,0.224,0.225]` | Channel-wise normalisation std. |
| `tampered_label` | int | `2` | Index used as the tampered class (segmentation). |
| `base_width` | int | `16` | Base channel width multiplier for the lightweight CNN/U-Net. |
| `backbone.name` | str | `"custom"` | Classification backbone identifier (only `custom` is wired up today). |
| `backbone.pretrained` | bool | `false` | Whether to load pretrained weights when using a torchvision backbone. |
| `backbone.trainable_layers` | int | `4` | Count of trainable layers when a torchvision backbone is selected. |

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
| `checkpoint_frequency` | int | `5` | Create a checkpoint every N epochs. |
| `keep_checkpoints` | int | `3` | Maximum number of checkpoints retained on disk. |
| `label_smoothing` | float | `0.0` | Apply label smoothing to cross-entropy (classification). |
| `grad_clip_norm` | float | `0.0` | Clip gradients to this L2 norm (0 disables clipping). |
| `ema_decay` | float | `0.0` | Exponential moving-average decay for model weights (0 disables). |
| `optimizer` | mapping | see below | Configure the optimiser family and hyperparameters. |
| `scheduler` | mapping | see below | Learning-rate schedule configuration. |
| `loss` | mapping | see below | Segmentation loss weighting (classification ignores it). |

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

### `training.scheduler`
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | str | `""` | Supported values: `""` (disabled), `"cosine"`, `"onecycle"`. |
| `t_max` | int | `0` | Override `T_max` for cosine annealing (defaults to `epochs`). |
| `max_lr` | float | `0.0` | Peak LR for OneCycle (defaults to `learning_rate`). |
| `pct_start` | float | `0.3` | Warm-up fraction for OneCycle. |
| `div_factor` | float | `25.0` | Initial LR divisor for OneCycle. |
| `final_div_factor` | float | `10000.0` | Final LR divisor for OneCycle. |

### `training.loss`
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `bce_weight` | float | `0.5` | Weight applied to the binary cross-entropy term in the segmentation loss. |
| `dice_weight` | float | `0.5` | Weight applied to the Dice loss term. |

---

## `paths`
Output locations. Paths are expanded relative to the repository root.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `base` | str | `outputs` | Root directory for run artefacts. |
| `task` | str | `""` | Filled in automatically by the CLI (`classification`/`segmentation`). |
| `run_id` | str | auto | Timestamp (`YYYYMMDDTHHMMSSZ`) generated at startup unless overridden with `--runid`. |
| `model_filename` | str | `best_model.pth` | Filename used when saving checkpoints. |
| `history_filename` | str | `history.json` | Training history file. |
| `metrics_filename` | str | `evaluation_metrics.json` | Evaluation results file. |
| `log_filename` | str | `logs` | Logs live under `<run_root>/logs/`. |
| `args_filename` | str | `args.json` | Stores CLI arguments for auditing. |
| `checkpoints` | str | `checkpoints` | Folder inside the run directory for saved checkpoints. |

---

## `evaluation`
Controls evaluation exports.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `report_background` | bool | `false` | Include background-only metrics for segmentation. |
| `background_samples` | int | `0` | Number of background samples to analyse when `report_background` is enabled. |
| `thresholds` | list[float] | `[0.1 … 0.9]` | Probability thresholds swept during metric calculation. |
| `bucket_edges` | list[float] | `[0.5, 2.0]` | Bin edges used when grouping examples by tamper size. |
| `analysis_examples` | int | `6` | Count of qualitative examples saved during segmentation evaluation. |

---

## Checklist When Editing Configs
1. Duplicate a template under `configs/` so you keep a record for each run.
2. Adjust the sections above—most experiments change `data`, `training`, and `training.optimizer`.
3. Point the CLI to your file with `--config`.
4. Keep output paths unique per run so results don’t overwrite each other.

That’s it: the dataclasses enforce types, defaults fill in the gaps, and any YAML override you supply becomes the source of truth for a run.
