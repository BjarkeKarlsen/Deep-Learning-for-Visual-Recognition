from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path


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
        # Cache the class count so downstream consumers skip recomputing len(class_names).
        self.num_classes = len(self.class_names)

@dataclass
class OptimizerConfig:
    name: str = "adam"
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9
    nesterov: bool = False

@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"  # ConfigLoader overwrites this based on accelerator availability
    save_interval: int = 5  # Save a checkpoint every N epochs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
@dataclass
class LossConfig:
    # Names of classification losses, e.g. ["crossentropy","focal"]
    cls_types: List[str] = field(default_factory=lambda: ["crossentropy"])
    # Corresponding weights
    cls_weights: List[float] = field(default_factory=lambda: [1.0])
    # Global kwargs applied to all classification losses
    cls_global_kwargs: Dict = field(default_factory=dict)
    # Per-loss overrides (list of dicts)
    cls_per_kwargs: List[Dict] = field(default_factory=list)

    # Names of segmentation losses, e.g. ["bce","dice"]
    seg_types: List[str] = field(default_factory=lambda: ["bce"])
    # Corresponding weights
    seg_weights: List[float] = field(default_factory=lambda: [1.0])
    # Global kwargs applied to all segmentation losses
    seg_global_kwargs: Dict = field(default_factory=dict)
    # Per-loss overrides (list of dicts)
    seg_per_kwargs: List[Dict] = field(default_factory=list)

@dataclass
class PathsConfig:
    base: Path = Path("outputs") 
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%SZ"))
    task: str = ""
    model_filename: str = "best_model.pth"
    history_filename: str = "history.json"
    metrics_filename: str = "metrics.json"
    log_filename: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%SZ") + ".log")
    args_filename: str = "args.json"
    
    def __post_init__(self):
        # Defer directory creation until the task name is populated.
        if self.task:
            (self.base / self.task / "runs" / self.run_id).mkdir(parents=True, exist_ok=True)
    
    @property
    def run_root(self) -> Path:
        return self.base / self.task / "runs" / self.run_id

    @property
    def model_path(self) -> Path:
        p = self.run_root / self.model_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def history_path(self) -> Path:
        p = self.run_root / self.history_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def metrics_path(self) -> Path:
        p = self.run_root / self.metrics_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        p = self.run_root / self.log_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def args_path(self) -> Path:
        p = self.run_root / self.args_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    
@dataclass(frozen=True)
class TaskConfig:
    CLASSIFICATION: str = "classification"
    SEGMENTATION: str = "segmentation"
    PLOT: str = "plot"
    
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    loss: LossConfig = field(default_factory=LossConfig)

