import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, ContextManager
from contextlib import contextmanager
import numpy as np
import pandas as pd


class SidLogger:
    """
    A comprehensive logger for training and evaluation workflows.
    Supports console+file outputs, structured metrics logging,
    evaluation report tables, and timing via context manager.
    """

    def __init__(
        self,
        name: str = "training",
        log_dir: Union[str, Path] = "logs",
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True,
        timestamp_fmt: str = "%Y%m%d_%H%M%S",
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.console = console
        self.file = file
        self.timestamp_fmt = timestamp_fmt
        self._logger: Optional[logging.Logger] = None
        self._setup()

    def _setup(self):
        if self._logger and self._logger.handlers:
            return

        if self.file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(self.name)
        self._logger.handlers.clear()
        self._logger.setLevel(self.level)

        # Detailed formatter for file, simple for console
        detailed = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        simple = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if self.file:
            ts = datetime.now().strftime(self.timestamp_fmt)
            path = self.log_dir / f"{self.name}_{ts}.log"
            fh = logging.FileHandler(path, mode="w")
            fh.setLevel(self.level)
            fh.setFormatter(detailed)
            self._logger.addHandler(fh)

        if self.console:
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(simple)
            self._logger.addHandler(ch)

        self._logger.propagate = False

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._setup()
        return self._logger

    def log_model_info(self, model, total_params: int, trainable_params: int):
        self.logger.info("=" * 60)
        self.logger.info("MODEL ARCHITECTURE")
        self.logger.info("=" * 60)
        self.logger.info(f"{model.__class__.__name__}")
        self.logger.info(f"Total params: {total_params:,}")
        self.logger.info(f"Trainable params: {trainable_params:,}")
        self.logger.info("=" * 60)

    def log_training_config(self, config: Dict[str, Any]):
        self.logger.info("=" * 60)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 60)
        for k, v in config.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("=" * 60)

    def log_evaluation_config(self, config: Dict[str, Any]):
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION CONFIGURATION")
        self.logger.info("=" * 60)
        for k, v in config.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("=" * 60)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        self.logger.info(f"Starting epoch {epoch+1}/{total_epochs}")

    def log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        msg = f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        if train_acc is not None:
            msg += f", train_acc={train_acc:.4f}"
        if val_acc is not None:
            msg += f", val_acc={val_acc:.4f}"
        if lr is not None:
            msg += f", lr={lr:.6f}"
        if duration is not None:
            msg += f", time={duration:.2f}s"
        self.logger.info(msg)

    def log_batch_progress(
        self, batch_idx: int, total_batches: int, loss: float, interval: int = 100
    ):
        if batch_idx % interval == 0 or batch_idx == total_batches:
            pct = 100 * batch_idx / total_batches
            self.logger.info(f"Batch {batch_idx}/{total_batches} ({pct:.1f}%) loss={loss:.6f}")

    def log_best_model(self, epoch: int, metric: str, value: float):
        self.logger.info(f"New best model at epoch {epoch+1}: {metric}={value:.4f}")

    def log_training_complete(
        self,
        *,
        total_time: Optional[float],
        best_metric: Optional[float],
        best_epoch: Optional[int],
    ):
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 60)
        if total_time is not None:
            self.logger.info(
                f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)"
            )
        else:
            self.logger.info("Total time: n/a")

        if best_metric is not None and best_epoch is not None:
            self.logger.info(f"Best metric {best_metric:.4f} at epoch {best_epoch+1}")
        else:
            self.logger.info("Best metric: n/a")
        self.logger.info("=" * 60)

    def log_classification_report(self, report: Dict[str, Any]):
        """
        Log sklearn classification_report dict as a table.
        """
        df = pd.DataFrame(report).transpose()
        self.logger.info("CLASSIFICATION REPORT")
        self.logger.info("\n" + df.to_string(float_format="{:.2f}".format))

    def log_confusion_matrix(self, cm: np.ndarray, labels: list):
        """
        Log confusion matrix with labels.
        """
        df = pd.DataFrame(cm, index=labels, columns=labels)
        self.logger.info("CONFUSION MATRIX")
        self.logger.info("\n" + df.to_string())

    def save_json(self, data: Any, filename: str, indent: int = 2):
        """
        Save any JSON-serializable data to a file under log_dir.
        """
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=indent)
        self.logger.info(f"Saved JSON to {path}")

    @contextmanager
    def time_block(self, name: str) -> ContextManager[None]:
        """
        Context manager for timing operations:
            with logger.time_block("load data"):
                ...
        Logs elapsed time on exit.
        """
        start = datetime.now()
        yield
        end = datetime.now()
        delta = (end - start).total_seconds()
        self.logger.info(f"{name} took {delta:.2f}s")

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)
