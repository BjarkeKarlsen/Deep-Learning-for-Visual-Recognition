import json
import logging
from datetime import datetime
from typing import Generic, Optional, Dict, List, Any, Union, TypeVar
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
from abc import ABC, abstractmethod

# TODO: Consider using pandas DataFrame for easier metric handling polars 
# TODO: Add support for TensorBoard logging
# TODO: Add support for different evaluation metrics (e.g., IoU for segmentation) and also for classification

@dataclass
class BaseEvaluationMetrics(ABC):
    """Base class for all evaluation metrics"""
    # SHARED FIELDS FOR CLASSIFICATION AND SEGMENTATION EVAL SUMMARIES.
    task_type: str
    primary_metric: str
    primary_score: float
    timestamp: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @abstractmethod
    def get_comparable_score(self) -> float:
        """Get the primary score for comparison across runs"""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics - can be overridden by subclasses"""
        return {
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "timestamp": self.timestamp
        }


@dataclass
class ClassificationEvaluationMetrics(BaseEvaluationMetrics):
    """Classification-specific evaluation metrics"""
    # STORES CLASSIFICATION ACCURACY PLUS SKLEARN REPORT ARTIFACTS.
    accuracy: float = 0.0
    classification_report: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        # Auto-set task_type and primary_metric if not provided
        if not hasattr(self, 'task_type') or not self.task_type:
            self.task_type = "classification"
        if not hasattr(self, 'primary_metric') or not self.primary_metric:
            self.primary_metric = "accuracy"
        if not hasattr(self, 'primary_score') or not self.primary_score:
            self.primary_score = self.accuracy
    
    def get_comparable_score(self) -> float:
        """Return accuracy as the comparable score"""
        return self.accuracy
    
    def get_summary(self) -> Dict[str, Any]:
        """Get classification-specific summary"""
        summary = super().get_summary()
        summary.update({
            "accuracy": self.accuracy,
            "num_classes": len(self.class_names),
            "macro_f1": self.classification_report.get("macro avg", {}).get("f1-score", 0.0),
            "weighted_f1": self.classification_report.get("weighted avg", {}).get("f1-score", 0.0)
        })
        return summary

@dataclass
class SegmentationEvaluationMetrics(BaseEvaluationMetrics):
    """Segmentation-specific evaluation metrics"""
    # STORES SEGMENTATION DICE/IOU SCORES AND OPTIONAL PER-CLASS DETAILS.
    mean_iou: float = 0.0
    pixel_accuracy: float = 0.0
    dice_coefficient: float = 0.0
    class_ious: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, 'task_type') or not self.task_type:
            self.task_type = "segmentation"
        if not hasattr(self, 'primary_metric') or not self.primary_metric:
            self.primary_metric = "mean_iou"
        if not hasattr(self, 'primary_score') or not self.primary_score:
            self.primary_score = self.mean_iou
    
    def get_comparable_score(self) -> float:
        """Return mean IoU as the comparable score"""
        return self.mean_iou
    
    def get_summary(self) -> Dict[str, Any]:
        """Get segmentation-specific summary"""
        summary = super().get_summary()
        summary.update({
            "mean_iou": self.mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "dice_coefficient": self.dice_coefficient,
            "num_classes": len(self.class_ious)
        })
        return summary
    

# Type variable for generic evaluation metrics tracker
T = TypeVar('T', bound=BaseEvaluationMetrics)

class IEvaluationTracker(ABC):
    """Interface for evaluation metrics tracking"""
    
    @abstractmethod
    def add_metrics(self, metrics: BaseEvaluationMetrics) -> None:
        """Add new evaluation metrics"""
        pass
    
    @abstractmethod
    def get_best_metric(self, metric_name: str = None) -> Optional[BaseEvaluationMetrics]:
        """Retrieve the best metric by name or primary metric"""
        pass
    
    @abstractmethod
    def save_to_json(self, path: Union[str, Path]) -> None:
        """Save metrics history to JSON file"""
        pass
    
    @abstractmethod
    def get_latest_metrics(self) -> Optional[BaseEvaluationMetrics]:
        """Get the most recent metrics"""
        pass

class EvaluationMetricsTracker(IEvaluationTracker, Generic[T]):
    """Universal evaluation metrics tracker that works with any task type"""
    
    def __init__(self, metric_type: type[T], logger: Optional[logging.Logger] = None):
        """
        Initialize tracker for specific metric type
        
        Args:
            metric_type: The specific evaluation metrics class to use
            logger: Optional logger for tracking operations
        """
        self.metric_type = metric_type
        self.metrics_history: List[T] = []
        self.lock = threading.RLock()
        self.logger = logger or logging.getLogger(__name__)
    
    def add_metrics(self, metrics: T) -> None:
        """Add new evaluation metrics"""
        if not isinstance(metrics, self.metric_type):
            raise TypeError(f"Expected {self.metric_type.__name__}, got {type(metrics).__name__}")
        
        with self.lock:
            # APPEND LATEST EVALUATION RESULTS TO THE HISTORY LIST.
            self.metrics_history.append(metrics)
            self.logger.info(f"Added {metrics.task_type} metrics: {metrics.primary_metric}={metrics.primary_score}")
    
    def get_best_metric(self, metric_name: str = None) -> Optional[T]:
        """
        Retrieve the best metric by name or primary metric
        
        Args:
            metric_name: Specific metric name to optimize for, or None for primary metric
        
        Returns:
            Best metrics object or None if no metrics exist
        """
        with self.lock:
            if not self.metrics_history:
                return None
            
            if metric_name is None:
                # Use primary score for comparison
                return max(self.metrics_history, key=lambda x: x.get_comparable_score())
            else:
                # Use specific metric from additional_metrics or main fields
                def get_metric_value(metrics: T) -> float:
                    # Try additional_metrics first
                    if metric_name in metrics.additional_metrics:
                        return metrics.additional_metrics[metric_name]
                    # Try main fields
                    if hasattr(metrics, metric_name):
                        return getattr(metrics, metric_name)
                    return float('-inf')
                
                return max(self.metrics_history, key=get_metric_value)
    
    def get_latest_metrics(self) -> Optional[T]:
        """Get the most recent metrics"""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def save_to_json(self, path: Union[str, Path]) -> None:
        """Save metrics history to JSON file"""
        with self.lock:
            # CONVERT PATH TO STRING IF NEEDED FOR JSON SERIALISATION.
            path = str(path)
            
            # CREATE THE DATA STRUCTURE THAT WILL BE WRITTEN TO DISK.
            data = {
                "task_type": self.metric_type.__name__,
                "total_evaluations": len(self.metrics_history),
                "created_at": datetime.now().isoformat(),
                "metrics_history": [asdict(metrics) for metrics in self.metrics_history]
            }
            
            # ADD SUMMARY STATISTICS FOR QUICK HUMAN INSPECTION.
            if self.metrics_history:
                latest = self.metrics_history[-1]
                best = self.get_best_metric()
                data["summary"] = {
                    "latest_metrics": latest.get_summary(),
                    "best_metrics": best.get_summary() if best else None,
                    "best_primary_score": best.get_comparable_score() if best else None
                }
            
            # WRITE THE JSON PAYLOAD TO DISK.
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=4, default=str)
                self.logger.info(f"Saved {len(self.metrics_history)} evaluation metrics to {path}")
            except Exception as e:
                self.logger.error(f"Failed to save metrics to {path}: {e}")
                raise
    
    def load_from_json(self, path: Union[str, Path]) -> None:
        """Load metrics history from JSON file"""
        with self.lock:
            path = str(path)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Load metrics history
                self.metrics_history = []
                for metrics_dict in data.get("metrics_history", []):
                    # Create instance of the appropriate metrics class
                    metrics = self.metric_type(**metrics_dict)
                    self.metrics_history.append(metrics)
                
                self.logger.info(f"Loaded {len(self.metrics_history)} evaluation metrics from {path}")
            except Exception as e:
                self.logger.error(f"Failed to load metrics from {path}: {e}")
                raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        with self.lock:
            if not self.metrics_history:
                return {
                    "total_evaluations": 0,
                    "task_type": self.metric_type.__name__
                }
            
            latest = self.metrics_history[-1]
            best = self.get_best_metric()
            
            return {
                "total_evaluations": len(self.metrics_history),
                "task_type": latest.task_type,
                "latest_metrics": latest.get_summary(),
                "best_metrics": best.get_summary() if best else None,
                "best_primary_score": best.get_comparable_score() if best else None
            }
