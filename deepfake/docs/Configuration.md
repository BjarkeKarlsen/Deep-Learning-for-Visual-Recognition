# CONFIGURATION GUIDE

This document outlines how configuration is structured and consumed across the
project.

---

## 1. Schema (`src/config/schema.py`)

Configuration is defined with `dataclasses`, giving type hints, defaults, and a
single canonical schema:

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
    tampered_label: int = 2

    def __post_init__(self) -> None:
        self.num_classes = len(self.class_names)


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"  # updated at runtime depending on CUDA availability


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

---

## 2. Baseline values (`src/config/default.yaml`)

`default.yaml` mirrors the schema and ships the repository’s baseline settings:

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

model:
  class_names:
    - Real
    - Synthetic
    - Tampered
  normalize_mean:
    - 0.485
    - 0.456
    - 0.406
  normalize_std:
    - 0.229
    - 0.224
    - 0.225
  tampered_label: 2

training:
  epochs: 100
  learning_rate: 0.001
  seed: 42

paths:
  model_path: models/best_model.pth
  results_dir: results
  history_file: training_history.json
  logging_dir: null
```

These defaults remain untouched at runtime and serve as documentation plus a
fallback for any missing keys in user overrides.

---

## 3. Override files (`classification.yaml`, `segmentation.yaml`, ...)

Override files live at the repository root and are passed explicitly via the
`--config` CLI flag. Example:

```bash
python main.py --train --task classification --config classification.yaml
python main.py --train --task segmentation   --config segmentation.yaml
```

Each override file can change any subset of keys. When a key is omitted it
falls back to `default.yaml` (and ultimately to dataclass defaults). You can
create additional configs—e.g. `configs/large-run.yaml`—and point `--config` at
that path.

---

## 4. Loading configuration (`src/utils/config_loader.py`)

`ConfigLoader` merges the schema + defaults + optional overrides and normalises
paths so scripts work regardless of the current working directory:

```python
from src.utils.config_loader import ConfigLoader

loader = ConfigLoader(config_path="classification.yaml")
cfg = loader.get_config()
```

Key behaviours:

- `OmegaConf.structured(Config)` initialises the schema with type safety.
- `default.yaml` is merged next.
- When `config_path` is provided, it is merged on top (the CLI requires this via
  `--config`).
- `cfg.training.device` is set to `"cuda"` when a GPU is available, otherwise
  `"cpu"`.
- `cfg.paths.model_path`, `cfg.paths.results_dir`, and (if set)
  `cfg.paths.logging_dir` are converted to absolute paths anchored to the repo
  root.
- The loader creates results and logging directories eagerly to avoid race
  conditions later in the pipeline.

### CLI helper

```bash
# Inspect the merged configuration
python -m src.utils.config_loader --config classification.yaml

# Dump the baseline template somewhere else
python -m src.utils.config_loader --dump-default --output my-config.yaml
# Add --force to overwrite an existing file
```

Together these pieces provide a predictable configuration story: dataclasses
establish the schema, `default.yaml` defines canonical defaults, override files
are explicit per-run knobs, and `ConfigLoader` ties everything together at
runtime.
