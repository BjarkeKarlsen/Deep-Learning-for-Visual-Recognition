# CONFIGURATION GUIDE

This document explains how the configuration system is organized after the refactor.

***

## 1. Schema (`src/config/schema.py`)

All configuration sections are defined with Python dataclasses:

```python
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset_name: str = "saberzl/SID_Set"
    image_size: int = 512
    train_samples: int = 10
    val_samples: int = 10
    test_samples: int = 10
    use_streaming: bool = True
    use_disk_cache: bool = True


@dataclass
class LoaderConfig:
    batch_size: int = 4
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    num_workers: int = 4


@dataclass
class ModelConfig:
    class_names: List[str] = field(default_factory=lambda: ["Real", "Synthetic", "Tampered"])
    num_classes: int = field(init=False)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def __post_init__(self) -> None:
        self.num_classes = len(self.class_names)


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"  # auto-filled at runtime


@dataclass
class PathsConfig:
    model_path: str = "models/best_model.pth"
    results_dir: str = "results"
    history_file: str = "training_history.json"
    logging_dir: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
```

The dataclasses provide types, defaults, and derived values (for example, `ModelConfig.num_classes`).

***

## 2. Default values (`src/config/default.yaml`)

A YAML file mirrors the dataclass structure and provides the baseline configuration shipped with the repository. A shortened view:

```yaml
# src/config/default.yaml

data:
  dataset_name: saberzl/SID_Set
  image_size: 512
  train_samples: 10
  val_samples: 10
  test_samples: 10
  use_streaming: true
  use_disk_cache: true

loader:
  batch_size: 4
  shuffle_train: true
  shuffle_val: false
  shuffle_test: false
  num_workers: 4

... (model, training, paths)
```

This file is never modified at runtime; it simply documents the shipped defaults.

***

## 3. User overrides (`config.yaml`)

Place your overrides in the repository root `config.yaml`. It already includes every available setting with the project’s preferred values, so you can edit directly without hunting for missing keys:

```yaml
# config.yaml (example)

data:
  dataset_name: saberzl/SID_Set
  image_size: 512
  train_samples: 100
  val_samples: 20
  test_samples: 10
  use_streaming: true
  use_disk_cache: true

loader:
  batch_size: 8
  shuffle_train: true
  shuffle_val: false
  shuffle_test: false
  num_workers: 4

... (model, training, paths)
```

Any key you omit falls back to the default YAML (and ultimately to the dataclass defaults if the key is missing there as well).

***

## 4. Loading configuration (`src/utils/config_loader.py`)

`ConfigLoader` handles the merge flow and resolves relative paths against the project root:

```python
from src.utils.config_loader import ConfigLoader

loader = ConfigLoader()            # looks for config.yaml in the repo root
cfg = loader.get_config()          # DictConfig backed by the dataclasses
```

Key behaviors:

- Defaults: `OmegaConf.structured(Config)` + `src/config/default.yaml`
- Overrides: merges `config.yaml` if it exists (you can pass a custom path)
- Device detection: automatically sets `cfg.training.device` to `cuda` when available
- Path normalization: `cfg.paths.model_path` and `cfg.paths.results_dir` become absolute paths anchored to the repository, so running scripts from other directories works seamlessly
- Directory preparation: results/logging directories are created on load

The module also exposes a small CLI helper:

```bash
# View the merged configuration
python -m src.utils.config_loader

# Write the default template to disk
python -m src.utils.config_loader --dump-default --output config.template.yaml
# Use --force if you need to overwrite an existing file
```

With this layout the configuration story is:

1. Dataclasses define structure and validation
2. `src/config/default.yaml` documents shipped defaults
3. `config.yaml` contains your experiment-specific overrides
4. `ConfigLoader` ties it all together at runtime

This separation keeps defaults discoverable, makes overrides explicit, and removes the need to run scripts from the repository root.
