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
    base_path: str = "outputs"
    model_path: str = "outputs/models/best_model.pth"
    results_dir: str = "outputs/results"
    history_file: str = "training_history.json"
    logging_dir: Optional[str] = "outputs/logs"


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
