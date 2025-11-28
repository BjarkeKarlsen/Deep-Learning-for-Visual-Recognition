"""User-facing helpers organised as small, composable steps.

You can chain the steps manually (load config → seed → create runtime →
build trainer → run training) or call the higher-level helpers at the bottom
of the file if you prefer a single function.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from deepfake.classification import ClassificationEvaluator, ClassificationTrainer
from deepfake.config import Config
from deepfake.segmentation import SegmentationEvaluator, SegmentationTrainer
from deepfake.utils.checkpoint_manager import CheckpointManager
from deepfake.utils.config_loader import ConfigLoader
from deepfake.utils.logger import SidLogger
from deepfake.utils.seed_manager import SeedManager
from deepfake.utils.training_metrics_tracker import TrainingMetricsTracker

@dataclass
class TrainingRuntime:
    """Holds the shared utilities a trainer expects."""

    logger: SidLogger
    metrics_tracker: TrainingMetricsTracker
    checkpoint_mgr: CheckpointManager
    verbose: bool


def load_config(config_path: Optional[str] = None) -> Config:
    """Resolve configuration using the same loader as the CLI."""
    loader = ConfigLoader(config_path=config_path)
    return loader.get_config()


def seed_everything(cfg: Config, *, seed: Optional[int] = None) -> None:
    """Seed RNG back-ends so subsequent steps are reproducible."""
    SeedManager(seed if seed is not None else getattr(cfg.training, "seed", 42))


def init_training_runtime(cfg: Config, logger_name: str, *, verbose: bool = True) -> TrainingRuntime:
    """Construct the logger, metrics tracker, and checkpoint manager."""
    logger = SidLogger(name=logger_name, log_dir=cfg.paths.log_path)

    metrics_tracker = TrainingMetricsTracker(logger=logger.logger)
    metrics_tracker.configure_backup(cfg.paths.history_path)

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoints_path,
        logger=logger.logger,
        backup_frequency=cfg.training.checkpoint_frequency,
        keep_checkpoints=cfg.training.keep_checkpoints,
    )
    if not verbose:
        logger.logger.setLevel(logging.WARNING)
    return TrainingRuntime(logger=logger, metrics_tracker=metrics_tracker, checkpoint_mgr=checkpoint_mgr, verbose=verbose)


def build_classification_trainer(cfg: Config, runtime: TrainingRuntime) -> ClassificationTrainer:
    """Initialise a classification trainer using a prepared runtime."""
    return ClassificationTrainer(
        cfg=cfg,
        logger=runtime.logger,
        metrics_tracker=runtime.metrics_tracker,
        checkpoint_mgr=runtime.checkpoint_mgr,
        verbose=runtime.verbose,
    )


def build_segmentation_trainer(cfg: Config, runtime: TrainingRuntime) -> SegmentationTrainer:
    """Initialise a segmentation trainer using a prepared runtime."""
    return SegmentationTrainer(
        cfg=cfg,
        logger=runtime.logger,
        metrics_tracker=runtime.metrics_tracker,
        checkpoint_mgr=runtime.checkpoint_mgr,
        verbose=runtime.verbose,
    )


def run_training(trainer, *, start_epoch: int = 0) -> None:
    """Kick off the training loop with optional resume epoch."""
    trainer.train(start_epoch=start_epoch)


def build_classification_evaluator(cfg: Config, *, logger_name: str = "classification-eval") -> ClassificationEvaluator:
    """Create an evaluator for the latest classification run described by ``cfg``."""
    logger = SidLogger(name=logger_name, log_dir=cfg.paths.log_path)
    return ClassificationEvaluator(cfg, logger)


def build_segmentation_evaluator(cfg: Config, *, logger_name: str = "segmentation-eval") -> SegmentationEvaluator:
    """Create an evaluator for the latest segmentation run described by ``cfg``."""
    logger = SidLogger(name=logger_name, log_dir=cfg.paths.log_path)
    return SegmentationEvaluator(cfg, logger)


def run_evaluation(evaluator) -> None:
    """Trigger evaluation for the supplied evaluator."""
    evaluator.run()


# --- High-level helpers built from the stepwise functions --------------------

def train_classification(config_path: Optional[str] = None, *, verbose: bool = True) -> Dict[str, Any]:
    """Train the classification model using a configuration file or the defaults."""
    cfg = load_config(config_path)
    seed_everything(cfg)
    runtime = init_training_runtime(cfg, "classification-train", verbose=verbose)
    trainer = build_classification_trainer(cfg, runtime)
    run_training(trainer)
    return _summarise_run(cfg, runtime.metrics_tracker)


def train_segmentation(config_path: Optional[str] = None, *, verbose: bool = True) -> Dict[str, Any]:
    """Train the segmentation model with optional quiet mode."""
    cfg = load_config(config_path)
    seed_everything(cfg)
    runtime = init_training_runtime(cfg, "segmentation-train", verbose=verbose)
    trainer = build_segmentation_trainer(cfg, runtime)
    run_training(trainer)
    return _summarise_run(cfg, runtime.metrics_tracker)


def evaluate_classification(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the latest classification run referenced by ``config_path``."""
    cfg = load_config(config_path)
    evaluator = build_classification_evaluator(cfg)
    run_evaluation(evaluator)
    return _collect_eval_artifacts(cfg.paths.metrics_path)


def evaluate_segmentation(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the latest segmentation run referenced by ``config_path``."""
    cfg = load_config(config_path)
    evaluator = build_segmentation_evaluator(cfg)
    run_evaluation(evaluator)
    return _collect_eval_artifacts(cfg.paths.metrics_path)


def _summarise_run(cfg: Config, metrics_tracker: TrainingMetricsTracker) -> Dict[str, Any]:
    """Return a small dictionary with run metadata for notebook-friendly introspection."""
    summary = metrics_tracker.get_summary_stats() or {}
    best = metrics_tracker.get_best_metric("val_acc") or metrics_tracker.get_best_metric("val_dice")

    payload: Dict[str, Any] = {
        "run_id": cfg.paths.run_id,
        "run_root": str(cfg.paths.run_root),
        "history_path": str(cfg.paths.history_path),
        "model_path": str(cfg.paths.model_path),
        "best_epoch": getattr(best, "epoch", None),
    }
    if summary:
        payload["summary"] = summary
    if best:
        best_dict = asdict(best)
        payload["best_metrics"] = best_dict
    return payload


def _collect_eval_artifacts(metrics_path: Path) -> Dict[str, Any]:
    """Load evaluation metrics JSON if it exists."""
    if not metrics_path.exists():
        return {"metrics_path": str(metrics_path), "metrics": None}
    import json

    with metrics_path.open() as fp:
        data = json.load(fp)
    return {
        "metrics_path": str(metrics_path),
        "metrics": data,
    }


def summarise_training(cfg: Config, runtime: TrainingRuntime) -> Dict[str, Any]:
    """Public helper that wraps the internal summary builder."""
    return _summarise_run(cfg, runtime.metrics_tracker)


def collect_evaluation(cfg: Config) -> Dict[str, Any]:
    """Fetch metrics emitted by a previous evaluation step."""
    return _collect_eval_artifacts(cfg.paths.metrics_path)
