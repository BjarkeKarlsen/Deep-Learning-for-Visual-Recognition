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
    #__slots__ = ['epoch', 'step', 'train_loss', 'val_loss', 'train_acc', 
    #             'val_acc', 'learning_rate', 'timestamp', 'additional_metrics']
    
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
        # Input validation
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
            
        # Set timestamp if not provided
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
    thread safety, and validation.
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 auto_backup: bool = True,
                 max_metrics_in_memory: int = 10000):
        """
        Initialize the metrics tracker.
        
        Args:
            logger: Logger instance for tracking operations
            auto_backup: Whether to automatically backup metrics
            max_metrics_in_memory: Maximum number of metrics to keep in memory
        """
        self.metrics: List[TrainingMetrics] = []
        self.best_metrics: Dict[str, TrainingMetrics] = {}
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.logger = logger or logging.getLogger(__name__)
        self.auto_backup = auto_backup
        self.max_metrics_in_memory = max_metrics_in_memory
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration
        self.config = {
            'track_train_loss': True,
            'track_val_loss': True,
            'track_train_acc': True,
            'track_val_acc': True,
            'track_learning_rate': True
        }
    
    def configure_tracking(self, **kwargs) -> None:
        """Configure which metrics to track."""
        with self._lock:
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
        Thread-safe implementation with memory management.
        """
        try:
            with self._lock:
                # Memory management
                if len(self.metrics) >= self.max_metrics_in_memory:
                    self.logger.warning(
                        f"Reached maximum metrics in memory ({self.max_metrics_in_memory}). "
                        f"Consider saving to disk."
                    )
                
                self.metrics.append(metrics)
                self._update_best_metrics(metrics)
                
                self.logger.debug(
                    f"Added metrics for epoch {metrics.epoch}, step {metrics.step}"
                )
                
        except Exception as e:
            self.logger.error(f"Error adding metrics: {e}")
            raise
    
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
                self.logger.debug(f"New best validation accuracy: {metrics.val_acc}")
        
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
                # Assume higher is better for custom metrics (can be configured)
                if (metric_name not in self.best_metrics or
                    value > self.best_metrics[metric_name].additional_metrics.get(metric_name, float('-inf'))):
                    self.best_metrics[metric_name] = metrics
                    self.logger.debug(f"New best {metric_name}: {value}")
    
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
                    'config': self.config.copy()
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
                if path.exists() and self.auto_backup:
                    backup_path = path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                    path.rename(backup_path)
                    self.logger.info(f"Created backup: {backup_path}")
                
                # Prepare data for serialization
                data = {
                    'metrics': [self._serialize_metrics(m) for m in self.metrics],
                    'best_metrics': {k: self._serialize_metrics(v) for k, v in self.best_metrics.items()},
                    'start_time': self.start_time,
                    'end_time': self.end_time,
                    'config': self.config,
                    'summary_stats': self.get_summary_stats(),
                    'version': '2.0',  # Version for backward compatibility
                    'created_at': datetime.now().isoformat()
                }
                
                # Write to temporary file first, then rename (atomic operation)
                temp_path = path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                temp_path.rename(path)
                self.logger.info(f"Successfully saved metrics to {path}")
                
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
                self.config.update(data.get('config', {}))
                
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
            self.logger.info("Reset all metrics and state")
    
    def get_metrics_subset(self, start_epoch: int = 0, end_epoch: Optional[int] = None) -> List[TrainingMetrics]:
        """Get a subset of metrics for a specific epoch range."""
        with self._lock:
            filtered_metrics = [
                m for m in self.metrics 
                if m.epoch >= start_epoch and (end_epoch is None or m.epoch <= end_epoch)
            ]
            return filtered_metrics
