from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from pathlib import Path

@dataclass
# AUGMENTATION SETTINGS CONTROL HOW IMAGES ARE RANDOMISED DURING TRAINING.
class AugmentationConfig:
    enable: bool = False
    random_resized_crop: bool = True
    scale_min: float = 0.8
    scale_max: float = 1.0
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.02
    gaussian_blur_prob: float = 0.0
    gaussian_blur_sigma_min: float = 0.1
    gaussian_blur_sigma_max: float = 2.0
    random_erasing_prob: float = 0.0
    random_erasing_scale_min: float = 0.02
    random_erasing_scale_max: float = 0.2
    random_erasing_ratio_min: float = 0.3
    random_erasing_ratio_max: float = 3.3
    preview_samples: int = 0
    preview_seed: int = 1234


@dataclass
# DATA SETTINGS DEFINE WHICH HUGGING FACE DATASET TO USE AND HOW MANY SAMPLES TO PULL.
class DataConfig:
    dataset_name: str = "saberzl/SID_Set"
    image_size: int = 512
    train_samples: int = 10
    val_samples: int = 10
    test_samples: int = 10
    use_streaming: bool = True
    use_disk_cache: bool = True
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)

@dataclass
# LOADER SETTINGS CONTROL PYTORCH DATALOADER BEHAVIOUR SUCH AS BATCH SIZE AND WORKERS.
class LoaderConfig:
    batch_size: int = 4
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = False

@dataclass
# BACKBONE SETTINGS SELECT WHICH FEATURE EXTRACTOR THE SEGMENTATION MODEL USES.
class BackboneConfig:
    name: str = "custom"
    pretrained: bool = False
    trainable_layers: int = 4

@dataclass
# EVALUATION SETTINGS CONFIGURE THRESHOLD SWEEPS, BUCKET REPORTS, AND OPTIONAL BACKGROUND CHECKS.
class EvaluationConfig:
    report_background: bool = False
    background_samples: int = 0
    thresholds: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bucket_edges: List[float] = field(default_factory=lambda: [0.5, 2.0])
    analysis_examples: int = 6

@dataclass
# MODEL SETTINGS COVER CLASS LABELS, NORMALISATION VALUES, AND BACKBONE PARAMETERS.
class ModelConfig:
    class_names: List[str] = field(default_factory=lambda: ["Real", "Synthetic", "Tampered"])
    num_classes: int = field(init=False)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    tampered_label: int = 2
    base_width: int = 16
    backbone: "BackboneConfig" = field(default_factory=lambda: BackboneConfig())

    def __post_init__(self) -> None:
        # CACHE THE CLASS COUNT SO DOWNSTREAM CODE DOES NOT RECALCULATE LEN(CLASS_NAMES).
        self.num_classes = len(self.class_names)

@dataclass
# OPTIMISER SETTINGS DESCRIBE WHICH OPTIMISATION ALGORITHM AND REGULARISATION TERMS TO USE.
class OptimizerConfig:
    name: str = "adam"
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9
    nesterov: bool = False

@dataclass
# SCHEDULER SETTINGS CONTROL OPTIONAL LEARNING RATE SCHEDULES.
class SchedulerConfig:
    name: str = ""
    t_max: int = 0
    max_lr: float = 0.0
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 10000.0

@dataclass
# LOSS SETTINGS TUNE THE RELATIVE WEIGHTS OF BCE AND DICE TERMS FOR SEGMENTATION.
class LossConfig:
    type: str = "bce"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

@dataclass
# TRAINING SETTINGS COVER GLOBAL HYPERPARAMETERS, DEVICE SELECTION, AND BOOK-KEEPING CADENCE.
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"  # CONFIG LOADER OVERWRITES THIS BASED ON ACCELERATOR AVAILABILITY.
    checkpoint_frequency: int = 5                # HOW OFTEN TO CREATE CHECKPOINTS.
    keep_checkpoints: int = 3             # HOW MANY CHECKPOINTS TO RETAIN ON DISK.
    label_smoothing: float = 0.0
    grad_clip_norm: float = 0.0
    ema_decay: float = 0.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

@dataclass
# PATH SETTINGS CONTROL WHERE RUN ARTIFACTS, LOGS, AND CHECKPOINTS ARE STORED.
class PathsConfig:
    base: Path = Path("outputs") 
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%SZ"))
    task: str = ""
    model_filename: str = "best_model.pth"
    history_filename: str = "history.json"
    metrics_filename: str = "metrics.json"
    log_filename: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%SZ") + ".log")
    args_filename: str = "args.json"
    checkpoints: str = "checkpoints"
    
    def __post_init__(self):
        # DEFER DIRECTORY CREATION UNTIL THE TASK NAME IS POPULATED.
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
    
    @property
    def checkpoints_path(self) -> Path:
        p = self.run_root / self.checkpoints
        p.mkdir(parents=True, exist_ok=True)
        return p
    
@dataclass(frozen=True)
# TASK ENUMERATION HELPS THE CLI MAP STRINGS TO KNOWN PIPELINES.
class TaskConfig:
    CLASSIFICATION: str = "classification"
    SEGMENTATION: str = "segmentation"
    PLOT: str = "plot"
    
@dataclass
# ROOT CONFIG OBJECT THAT BUNDLES ALL SUB-CONFIGS INTO ONE STRUCTURE.
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    Task: TaskConfig = field(default_factory=TaskConfig)
