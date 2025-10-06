from dataclasses import dataclass, field
from typing import List, Optional
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
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"  # ConfigLoader overwrites this based on accelerator availability
    save_interval: int = 5  # Save a checkpoint every N epochs


@dataclass
class PathsConfig:
    base: Path = Path("outputs")
    env: str = "dev"
    task: str = "segmentation"
    model_filename: str = "best_model.pth"
    history_filename: str = "history.json"
    eval_metrics_filename: str = "evaluation_metrics.json"
    log_subdir: str = "logs"

    def __post_init__(self):
        self.base.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        return self.base / "models" / self.task / self.env 

    @property
    def model_path(self) -> Path:
        p = self.model_dir / self.model_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir(self) -> Path:
        p = self.base / "results" / self.task / self.env
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def history_path(self) -> Path:
        return self.output_dir / self.history_filename

    @property
    def eval_metrics_path(self) -> Path:
        """
        Path to save/load evaluation metrics JSON.
        """
        return self.output_dir / self.eval_metrics_filename

    @property
    def log_dir(self) -> Path:
        p = self.base / self.log_subdir / self.task / self.env
        p.mkdir(parents=True, exist_ok=True)
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
    Task: TaskConfig = field(default_factory=TaskConfig)
