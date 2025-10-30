
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
from abc import ABC, abstractmethod


@dataclass
class TrainingMetrics:
    """
    Data class for training metrics at specific step/epoch.

    Uses slots for memory optimization and includes validation.
    """
    # CAPTURES ONE POINT IN THE TRAINING CURVE WITH OPTIONAL EXTRA METRICS.
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: Optional[str] = None
    additional_metrics: Optional[Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate inputs and set default timestamp."""
        # INPUT VALIDATION ENSURES METRICS STAY WITHIN EXPECTED RANGES.
        if self.epoch < 0:
            raise ValueError("Epoch must be non-negative")
        if self.step < 0:
            raise ValueError("Step must be non-negative")
        if self.train_loss < 0:
            raise ValueError("Training loss must be non-negative")
        if self.val_loss is not None and self.val_loss < 0:
            raise ValueError("Validation loss must be non-negative")
        if self.train_acc is not None and not (0 <= self.train_acc <= 1):
            raise ValueError("Training accuracy must be between 0 and 1")
        if self.val_acc is not None and not (0 <= self.val_acc <= 1):
            raise ValueError("Validation accuracy must be between 0 and 1")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        # SET TIMESTAMP IF NOT PROVIDED SO METRIC HISTORY IS TIME-STAMPED.
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class IMetricsTracker(ABC):
    """Interface for tracking training metrics with enhanced type hints."""

    @abstractmethod
    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add new training metrics to the tracker."""
        pass

    @abstractmethod
    def get_best_metric(self, metric_name: str) -> Optional[TrainingMetrics]:
        """Retrieve the best metric by name."""
        pass

    @abstractmethod
    def get_training_curves_data(self) -> Dict[str, List[Any]]:
        """Get training data in format suitable for plotting."""
        pass

    @abstractmethod
    def save_to_json(self, path: Union[str, Path]) -> None:
        """Save training history to JSON file."""
        pass


class TrainingMetricsTracker(IMetricsTracker):
    """
    Enhanced implementation of metrics tracking with improved error handling,
    thread safety, validation, and automatic backup functionality.
    """
    # MAINTAINS TRAINING HISTORY, BEST SCORES, AND OPTIONAL BACKUPS.

    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 auto_backup: bool = True,
                 max_metrics_in_memory: int = 10000,
                 backup_frequency: int = 5,
                 keep_backups: int = 5):
        """
        Initialize the metrics tracker.

        Args:
            logger: Logger instance for tracking operations
            auto_backup: Whether to automatically backup metrics
            max_metrics_in_memory: Maximum number of metrics to keep in memory
            backup_frequency: Create backup every N epochs (default: 5)
            keep_backups: Number of backup files to keep (default: 5)
        """
        self.metrics: List[TrainingMetrics] = []
        self.best_metrics: Dict[str, TrainingMetrics] = {}
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.logger = logger or logging.getLogger(__name__)
        self.auto_backup = auto_backup
        self.max_metrics_in_memory = max_metrics_in_memory

        # NEW: BACKUP CONFIGURATION TRACKS HOW OFTEN AND HOW MANY BACKUPS TO KEEP.
        self.backup_frequency = backup_frequency
        self.keep_backups = keep_backups
        self._backup_base_path: Optional[Path] = None

        # THREAD SAFETY: USE A REENTRANT LOCK BECAUSE TRACKER MAY BE CALLED FROM DIFFERENT THREADS.
        self._lock = threading.RLock()

        # CONFIGURATION DICT MIRRORS WHAT FIELDS ARE BEING TRACKED AND HOW OFTEN TO BACKUP.
        self.config = {
            'track_train_loss': True,
            'track_val_loss': True,
            'track_train_acc': True,
            'track_val_acc': True,
            'track_learning_rate': True,
            'backup_frequency': backup_frequency,
            'keep_backups': keep_backups
        }
        self.metric_directions: Dict[str, str] = {}

    def configure_backup(self, backup_base_path: Union[str, Path]) -> None:
        # CHOOSE WHERE PERIODIC BACKUP FILES ARE WRITTEN.
        """
        Configure the base path for backup files.

        Args:
            backup_base_path: Base path where backup files will be stored
        """
        with self._lock:
            self._backup_base_path = Path(backup_base_path)
            self._backup_base_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Configured backup base path: {self._backup_base_path}")

            # Load full history (metrics + best_metrics) if it exists
            if self._backup_base_path.is_file():
                try:
                    self.load_from_json(self._backup_base_path)
                except Exception as e:
                    self.logger.error(f"Failed to load backup history: {e}")
                    # Fallback to empty state
                    self.metrics = []
                    self.best_metrics = {}
            else:
                self.metrics = []
                self.best_metrics = {}

    def configure_tracking(self, **kwargs) -> None:
        """Configure which metrics to track and backup settings."""
        with self._lock:
            # UPDATE BACKUP SETTINGS IF PROVIDED BY THE CALLER.
            if 'backup_frequency' in kwargs:
                self.backup_frequency = kwargs['backup_frequency']
            if 'keep_backups' in kwargs:
                self.keep_backups = kwargs['keep_backups']

            metric_directions = kwargs.pop('metric_directions', None)
            if metric_directions is not None:
                self.metric_directions = {
                    name: direction.lower()
                    for name, direction in metric_directions.items()
                }
                if self.metric_directions:
                    self.config['metric_directions'] = self.metric_directions.copy()
                elif 'metric_directions' in self.config:
                    self.config.pop('metric_directions')

            self.config.update(kwargs)
            self.logger.info(f"Updated tracking configuration: {self.config}")

    def start_training(self) -> None:
        """Record the start time of training."""
        with self._lock:
            self.start_time = datetime.now().isoformat()
            self.logger.info(f"Training started at {self.start_time}")

    def end_training(self) -> None:
        """Record the end time of training."""
        with self._lock:
            self.end_time = datetime.now().isoformat()
            self.logger.info(f"Training ended at {self.end_time}")

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """
        Add new training metrics and update best metrics.
        Thread-safe implementation with memory management and automatic backup.
        """
        try:
            with self._lock:
                # MEMORY MANAGEMENT: WARN IF THE IN-MEMORY HISTORY GROWS BEYOND THE SAFETY LIMIT.
                if len(self.metrics) >= self.max_metrics_in_memory:
                    self.logger.warning(
                        f"Reached maximum metrics in memory ({self.max_metrics_in_memory}). "
                        f"Consider saving to disk."
                    )

                self.metrics.append(metrics)
                self._update_best_metrics(metrics)

                # Automatic backup every N epochs
                # PERIODICALLY WRITE BACKUP COPIES OF THE HISTORY FILE.
                if (self.auto_backup and 
                    self._backup_base_path and 
                    self.backup_frequency > 0 and
                    metrics.epoch > 0 and
                    metrics.epoch % self.backup_frequency == 0):
                    self._create_epoch_backup(metrics.epoch)

                self.logger.debug(
                    f"Added metrics for epoch {metrics.epoch}, step {metrics.step}"
                )

        except Exception as e:
            self.logger.error(f"Error adding metrics: {e}")
            raise

    def _create_epoch_backup(self, epoch: int) -> None:
        """
        Create a backup of metrics at a specific epoch.

        Args:
            epoch: The current epoch number
        """
        try:
            if not self._backup_base_path:
                self.logger.warning("Backup base path not configured, skipping backup")
                return

            # Create backup file with epoch suffix
            backup_path = self._backup_base_path.with_suffix(f'.epoch_{epoch:03d}.json')

            # Save current metrics to backup
            self.save_to_json(backup_path)

            # Clean up old backups
            self._cleanup_old_backups()

            self.logger.info(f"Created epoch {epoch} backup: {backup_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to create epoch {epoch} backup: {e}")
            # Don't raise exception - backup failure shouldn't stop training

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files, keeping only the latest N backups."""
        try:
            if not self._backup_base_path:
                return

            # Find all backup files with the pattern
            backup_dir = self._backup_base_path.parent
            backup_pattern = f"{self._backup_base_path.stem}.epoch_*.json"

            import glob
            backup_files = glob.glob(str(backup_dir / backup_pattern))

            if len(backup_files) <= self.keep_backups:
                return

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)

            # Remove oldest backups
            for backup_file in backup_files[self.keep_backups:]:
                Path(backup_file).unlink()
                self.logger.info(f"Cleaned up old backup: {Path(backup_file).name}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")

    def get_backup_files(self) -> List[Path]:
        """
        Get a list of all available backup files.

        Returns:
            List of backup file paths sorted by epoch number
        """
        try:
            if not self._backup_base_path:
                return []

            backup_dir = self._backup_base_path.parent
            backup_pattern = f"{self._backup_base_path.stem}.epoch_*.json"

            import glob
            backup_files = [Path(f) for f in glob.glob(str(backup_dir / backup_pattern))]

            # Sort by epoch number extracted from filename
            def extract_epoch(path: Path) -> int:
                try:
                    # Extract epoch number from filename like "history.epoch_005.json"
                    epoch_part = path.stem.split('.')[-1]  # "epoch_005"
                    return int(epoch_part.split('_')[1])    # 5
                except:
                    return 0

            backup_files.sort(key=extract_epoch)
            return backup_files

        except Exception as e:
            self.logger.error(f"Error getting backup files: {e}")
            return []

    def load_from_backup(self, epoch: int) -> bool:
        """
        Load metrics from a specific epoch backup.

        Args:
            epoch: The epoch number to load from

        Returns:
            True if backup was successfully loaded, False otherwise
        """
        try:
            if not self._backup_base_path:
                self.logger.error("Backup base path not configured")
                return False

            backup_path = self._backup_base_path.with_suffix(f'.epoch_{epoch:03d}.json')

            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False

            self.load_from_json(backup_path)
            self.logger.info(f"Successfully loaded from epoch {epoch} backup")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load from epoch {epoch} backup: {e}")
            return False

    def _update_best_metrics(self, metrics: TrainingMetrics) -> None:
        """Update best metrics based on new metrics with configuration."""

        # Update validation loss (lower is better)
        if (metrics.val_loss is not None and 
            self.config.get('track_val_loss', True)):
            if ('val_loss' not in self.best_metrics or 
                metrics.val_loss < self.best_metrics['val_loss'].val_loss):
                self.best_metrics['val_loss'] = metrics
                self.logger.debug(f"New best validation loss: {metrics.val_loss}")

        # Update training loss (lower is better)
        if (metrics.train_loss is not None and 
            self.config.get('track_train_loss', True)):
            if ('train_loss' not in self.best_metrics or 
                metrics.train_loss < self.best_metrics['train_loss'].train_loss):
                self.best_metrics['train_loss'] = metrics
                self.logger.debug(f"New best training loss: {metrics.train_loss}")

        # Update validation accuracy (higher is better)
        if (metrics.val_acc is not None and 
            self.config.get('track_val_acc', True)):
            if ('val_acc' not in self.best_metrics or 
                metrics.val_acc > self.best_metrics['val_acc'].val_acc):
                self.best_metrics['val_acc'] = metrics
                self.logger.info(f"New best validation accuracy: {metrics.val_acc:.4f}")

        # Update training accuracy (higher is better)
        if (metrics.train_acc is not None and 
            self.config.get('track_train_acc', True)):
            if ('train_acc' not in self.best_metrics or 
                (self.best_metrics['train_acc'].train_acc is None or 
                 metrics.train_acc > self.best_metrics['train_acc'].train_acc)):
                self.best_metrics['train_acc'] = metrics
                self.logger.debug(f"New best training accuracy: {metrics.train_acc}")

        # Handle additional metrics dynamically
        if metrics.additional_metrics:
            for metric_name, value in metrics.additional_metrics.items():
                if value is None:
                    continue

                direction = self._get_metric_direction(metric_name)
                if metric_name not in self.best_metrics:
                    self.best_metrics[metric_name] = metrics
                    self.logger.info(f"Tracking new metric {metric_name} ({direction}) with value {value}")
                    continue

                previous = self.best_metrics[metric_name].additional_metrics.get(metric_name)

                if previous is None:
                    self.best_metrics[metric_name] = metrics
                    self.logger.debug(f"Set baseline for {metric_name}: {value}")
                    continue

                if direction == 'min':
                    if value < previous:
                        self.best_metrics[metric_name] = metrics
                        self.logger.info(f"New best (min) {metric_name}: {value}")
                elif direction == 'last':
                    self.best_metrics[metric_name] = metrics
                    self.logger.debug(f"Updated {metric_name} to latest value {value}")
                else:  # default to max
                    if value > previous:
                        self.best_metrics[metric_name] = metrics
                        self.logger.info(f"New best (max) {metric_name}: {value}")

    def _get_metric_direction(self, metric_name: str) -> str:
        """Determine whether a metric should be maximised, minimised, or just tracked."""
        if metric_name in self.metric_directions:
            direction = self.metric_directions[metric_name]
            if direction in {'min', 'max', 'last'}:
                return direction

        name = metric_name.lower()
        loss_keywords = ('loss', 'error', 'mae', 'mse', 'rmse', 'perplexity', 'nll')
        maximise_keywords = ('acc', 'accuracy', 'dice', 'iou', 'precision', 'recall', 'f1', 'auc', 'psnr', 'ssim', 'tpr', 'tnr')

        if any(keyword in name for keyword in loss_keywords):
            return 'min'
        if 'lr' in name or 'learning_rate' in name:
            return 'last'
        if any(keyword in name for keyword in maximise_keywords):
            return 'max'

        # Default to maximising if direction is unknown
        return 'max'

    def get_best_metric(self, metric_name: str) -> Optional[TrainingMetrics]:
        """Retrieve the best metric by key with thread safety."""
        with self._lock:
            return self.best_metrics.get(metric_name)

    def get_training_curves_data(self) -> Dict[str, List[Any]]:
        """
        Get training data in format suitable for plotting.
        Enhanced with better filtering and error handling.
        """
        with self._lock:
            try:
                data = {
                    'epochs': [m.epoch for m in self.metrics],
                    'steps': [m.step for m in self.metrics],
                    'timestamps': [m.timestamp for m in self.metrics]
                }

                # Add metrics based on availability and configuration
                if self.config.get('track_train_loss', True):
                    data['train_loss'] = [m.train_loss for m in self.metrics]

                if self.config.get('track_val_loss', True):
                    data['val_loss'] = [m.val_loss for m in self.metrics if m.val_loss is not None]

                if self.config.get('track_train_acc', True):
                    data['train_acc'] = [m.train_acc for m in self.metrics if m.train_acc is not None]

                if self.config.get('track_val_acc', True):
                    data['val_acc'] = [m.val_acc for m in self.metrics if m.val_acc is not None]

                if self.config.get('track_learning_rate', True):
                    data['learning_rate'] = [m.learning_rate for m in self.metrics if m.learning_rate is not None]

                # Add additional metrics
                additional_metric_names = set()
                for m in self.metrics:
                    if m.additional_metrics:
                        additional_metric_names.update(m.additional_metrics.keys())

                for metric_name in additional_metric_names:
                    data[metric_name] = [
                        m.additional_metrics.get(metric_name) 
                        for m in self.metrics 
                        if m.additional_metrics and metric_name in m.additional_metrics
                    ]

                return data

            except Exception as e:
                self.logger.error(f"Error getting training curves data: {e}")
                raise

    def get_last_epoch(self) -> int:
        """Get the last completed epoch number with thread safety."""
        with self._lock:
            if not self.metrics:
                return -1
            return max(m.epoch for m in self.metrics)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics of the training history.
        Enhanced with more detailed statistics.
        """
        with self._lock:
            if not self.metrics:
                return {}

            try:
                stats = {
                    'total_metrics_recorded': len(self.metrics),
                    'last_epoch': self.get_last_epoch(),
                    'total_steps': max(m.step for m in self.metrics),
                    'duration_minutes': None,
                    'config': self.config.copy(),
                    'backup_files': len(self.get_backup_files())
                }

                # Calculate training duration
                if self.start_time and self.end_time:
                    try:
                        start = datetime.fromisoformat(self.start_time)
                        end = datetime.fromisoformat(self.end_time)
                        duration = (end - start).total_seconds() / 60
                        stats['duration_minutes'] = round(duration, 2)
                    except Exception as e:
                        self.logger.warning(f"Could not calculate duration: {e}")

                # Add best metric values with more details
                for metric_name, best_metric in self.best_metrics.items():
                    metric_value = getattr(best_metric, metric_name, 
                                         best_metric.additional_metrics.get(metric_name))
                    stats[f'best_{metric_name}'] = metric_value
                    stats[f'best_{metric_name}_epoch'] = best_metric.epoch
                    stats[f'best_{metric_name}_step'] = best_metric.step
                    stats[f'best_{metric_name}_timestamp'] = best_metric.timestamp

                # Add statistical summaries
                if self.metrics:
                    train_losses = [m.train_loss for m in self.metrics]
                    stats['train_loss_mean'] = sum(train_losses) / len(train_losses)
                    stats['train_loss_min'] = min(train_losses)
                    stats['train_loss_max'] = max(train_losses)

                return stats

            except Exception as e:
                self.logger.error(f"Error calculating summary stats: {e}")
                raise

    def save_to_json(self, path: Union[str, Path]) -> None:
        """
        Save training history to JSON file with enhanced error handling and backup.
        """
        path = Path(path)

        try:
            with self._lock:
                # Create directory if it doesn't exist
                path.parent.mkdir(parents=True, exist_ok=True)

                # Create backup if file exists and auto_backup is enabled
                if path.exists() and self.auto_backup and not path.name.startswith('.'):
                    backup_path = path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                    path.rename(backup_path)
                    self.logger.debug(f"Created backup: {backup_path}")

                # Prepare data for serialization
                data = {
                    'metrics': [self._serialize_metrics(m) for m in self.metrics],
                    'best_metrics': {k: self._serialize_metrics(v) for k, v in self.best_metrics.items()},
                    'start_time': self.start_time,
                    'end_time': self.end_time,
                    'config': self.config,
                    'summary_stats': self.get_summary_stats(),
                    'version': '2.1',  # Updated version with backup support
                    'created_at': datetime.now().isoformat()
                }

                # Write to temporary file first, then rename (atomic operation)
                temp_path = path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

                temp_path.rename(path)
                self.logger.debug(f"Successfully saved metrics to {path}")

        except Exception as e:
            self.logger.error(f"Error saving metrics to {path}: {e}")
            raise

    def _serialize_metrics(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Safely serialize TrainingMetrics to dict."""
        try:
            # Convert to dict, handling potential serialization issues
            metrics_dict = asdict(metrics)

            # Ensure all values are JSON serializable
            for key, value in metrics_dict.items():
                if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                    metrics_dict[key] = str(value)

            return metrics_dict

        except Exception as e:
            self.logger.error(f"Error serializing metrics: {e}")
            raise

    def load_from_json(self, path: Union[str, Path]) -> None:
        """
        Load training history from JSON file with validation.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")

        try:
            with self._lock:
                with open(path, 'r') as f:
                    data = json.load(f)

                # Version compatibility check
                version = data.get('version', '1.0')
                self.logger.info(f"Loading metrics from version {version}")

                # Load metrics
                self.metrics = []
                for metrics_data in data.get('metrics', []):
                    try:
                        # Handle potential missing fields for backward compatibility
                        metrics = TrainingMetrics(**metrics_data)
                        self.metrics.append(metrics)
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid metrics entry: {e}")

                # Load best metrics
                self.best_metrics = {}
                for name, metrics_data in data.get('best_metrics', {}).items():
                    try:
                        metrics = TrainingMetrics(**metrics_data)
                        self.best_metrics[name] = metrics
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid best metric {name}: {e}")

                # Load other data
                self.start_time = data.get('start_time')
                self.end_time = data.get('end_time')
                loaded_config = data.get('config', {})

                # Preserve current backup configuration when loading
                backup_config = {
                    'backup_frequency': self.backup_frequency,
                    'keep_backups': self.keep_backups
                }
                self.config.update(loaded_config)
                metric_directions = self.config.get('metric_directions', {})
                if isinstance(metric_directions, dict):
                    self.metric_directions = {
                        name: direction.lower()
                        for name, direction in metric_directions.items()
                    }
                self.config.update(backup_config)

                self.logger.info(f"Successfully loaded {len(self.metrics)} metrics from {path}")

        except Exception as e:
            self.logger.error(f"Error loading metrics from {path}: {e}")
            raise

    def reset(self) -> None:
        """Reset all tracked metrics and state."""
        with self._lock:
            self.metrics.clear()
            self.best_metrics.clear()
            self.start_time = None
            self.end_time = None
            self.metric_directions.clear()
            self.logger.info("Reset all metrics and state")

    def get_metrics_subset(self, start_epoch: int = 0, end_epoch: Optional[int] = None) -> List[TrainingMetrics]:
        """Get a subset of metrics for a specific epoch range."""
        with self._lock:
            filtered_metrics = [
                m for m in self.metrics 
                if m.epoch >= start_epoch and (end_epoch is None or m.epoch <= end_epoch)
            ]
            return filtered_metrics
