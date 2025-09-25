# CONFIGURATION GUIDE

This document explains how **`config.py`** and **`config.yaml`** work together to control all aspects of the project.

***

## 1. config.py

Defines the schema of all configurable settings using Python dataclasses. There are five sections:

```python
# config.py

@dataclass
class DataConfig:
    dataset_name:   str  = "saberzl/SID_Set"
    train_samples:  int  = 10
    val_samples:    int  = 10
    test_samples:   int  = 10
    use_streaming:  bool = True
    use_disk_cache: bool = True

@dataclass
class LoaderConfig:
    batch_size:    int  = 4
    shuffle_train: bool = True
    shuffle_val:   bool = False
    shuffle_test:  bool = False
    num_workers:   int  = 4

@dataclass
class ModelConfig:
    num_classes:   int          = 3
    class_names:   dict         = field(default_factory=lambda: {0:'Real',1:'Synthetic',2:'Tampered'})
    image_size:    int          = 512
    normalize_mean: List[float] = field(default_factory=lambda: [0.485,0.456,0.406])
    normalize_std:  List[float] = field(default_factory=lambda: [0.229,0.224,0.225])

@dataclass
class TrainingConfig:
    epochs:        int    = 100
    learning_rate: float  = 0.001
    seed:          int    = 42
    device:        str    = "cpu"   # auto-set at runtime

@dataclass
class PathsConfig:
    model_path:   str           = "models/best_model.pth"
    results_dir:  str           = "results"
    history_file: str           = "training_history.json"
    logging_dir:  Optional[str] = None

@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    loader:   LoaderConfig   = field(default_factory=LoaderConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths:    PathsConfig    = field(default_factory=PathsConfig)
```

- **DataConfig**: Dataset name, sample limits, caching behavior.
- **LoaderConfig**: DataLoader parameters (batch size, shuffle flags, workers).
- **ModelConfig**: Network parameters and normalization constants.
- **TrainingConfig**: Training loop hyperparameters and random seed.
- **PathsConfig**: File paths for model checkpoints, results, logging.
- **Config**: Root object aggregating all sections.

***

## 2. config.yaml

A YAML file to **override** any defaults from `config.py` without editing code. Place it in the project root:

```yaml
# config.yaml

data:
  train_samples: 100
  val_samples: 20

loader:
  batch_size: 8
  shuffle_train: true
  shuffle_val: false
  shuffle_test: false
  num_workers: 4

training:
  epochs: 50
  learning_rate: 0.0005

paths:
  logging_dir: "logs/exp1"
```

- Keys correspond to sections in `Config`.  
- Omitted keys use the defaults from `config.py`.  
- Any valid YAML boolean, string, number is parsed into the typed config.

***

## 3. Loading Configuration

Use `OmegaConf` to merge defaults and overrides:

```python
from omegaconf import OmegaConf
from config import Config

def load_config(path="config.yaml") -> Config:
    base = OmegaConf.structured(Config)
    if os.path.exists(path):
        overrides = OmegaConf.load(path)
        cfg = OmegaConf.merge(base, overrides)
    else:
        cfg = base
    # auto-detect device
    import torch
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg
```

- **Structured Config** ensures type safety.  
- **Merge** applies YAML overrides on top of dataclass defaults.  
- **Device detection** sets `cfg.training.device` at runtime.

***

## 4. Using the Config

Pass the loaded `cfg` everywhere:

```python
cfg = load_config("config.yaml")

# Access example:
print(cfg.data.train_samples)
print(cfg.loader.batch_size)
print(cfg.model.image_size)
print(cfg.training.epochs)
print(cfg.paths.results_dir)
```

In training script:

```python
train_loader = DataLoader(
    SIDDataset(...),
    batch_size=cfg.loader.batch_size,
    shuffle=cfg.loader.shuffle_train,
    num_workers=cfg.loader.num_workers,
)
```

In model creation:

```python
model = SimpleCNN(num_classes=cfg.model.num_classes).to(cfg.training.device)
```

Paths usage:

```python
os.makedirs(cfg.paths.results_dir, exist_ok=True)
torch.save(model.state_dict(), cfg.paths.model_path)
```

This unified approach centralizes all parameters, makes experiments reproducible, and keeps code clean.