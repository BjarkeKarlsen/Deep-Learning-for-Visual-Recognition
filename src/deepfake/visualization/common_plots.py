import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Any, Dict, Optional, Union

class CommonPlots:
    """Base class for all plotting functionality with common utilities."""

    def __init__(self, output_dir: str = None, load_training: str = None, load_evaluation: str = None):
        self.output_path = output_dir
        self.load_training = load_training
        self.load_evaluation = load_evaluation
        self.default_save_path = Path("results")
        
        
        # Set up default styling
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Set up consistent plotting style across all plots."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'grid.alpha': 0.3
        })
    
    def _resolve_path(self, save_path: str = None, default_filename: str = "plot.png") -> Path:
        """Build the final path using the provided save_path or instance save_path_dir."""
        if save_path:
            return Path(save_path)

         # Use instance save_path or fallback to default
        base_dir = Path(self.output_path) if self.output_path else self.default_save_path
        return base_dir / default_filename
    
    def _create_save_path(self, path: Path) -> Path:
        """Create save_path directory if it doesn't exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent
    
    def load_training_history(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load training history from TrainingMetricsTracker JSON format.
        
        Args:
            json_path: Path to the training history JSON file
            
        Returns:
            Dictionary with processed training curves data
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Training history file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract metrics from the tracker format
        metrics = data.get('metrics', [])
        
        if not metrics:
            raise ValueError("No metrics found in the training history file")
        
        # Initialize curves dictionary
        curves = {
            'epochs': [],
            'steps': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'timestamps': []
        }
        
        # Track additional metrics dynamically
        additional_metric_names = set()
        
        # Process each metric entry
        for metric in metrics:
            curves['epochs'].append(metric['epoch'])
            curves['steps'].append(metric['step'])
            curves['train_loss'].append(metric['train_loss'])
            curves['val_loss'].append(metric.get('val_loss'))
            curves['train_acc'].append(metric.get('train_acc'))
            curves['val_acc'].append(metric.get('val_acc'))
            curves['learning_rate'].append(metric.get('learning_rate'))
            curves['timestamps'].append(metric.get('timestamp'))
            
            # Handle additional metrics
            additional_metrics = metric.get('additional_metrics')
            if additional_metrics:
                for key, value in additional_metrics.items():
                    if key not in curves:
                        curves[key] = [None] * len(curves['epochs'])  # Fill previous entries with None
                        additional_metric_names.add(key)
                    curves[key][-1] = value  # Set current value
            
            # Ensure all additional metrics have entries for this step
            for key in additional_metric_names:
                if key not in curves:
                    curves[key] = [None] * len(curves['epochs'])
                elif len(curves[key]) < len(curves['epochs']):
                    curves[key].append(None)
        
        # Clean up None values for plotting
        processed_curves = {}
        for key, values in curves.items():
            if key in ['epochs', 'steps', 'timestamps']:
                processed_curves[key] = values  # Keep these as lists
            else:
                # Filter out None values but keep track of original indices
                filtered_values = []
                filtered_epochs = []
                for i, value in enumerate(values):
                    if value is not None:
                        filtered_values.append(value)
                        if i < len(curves['epochs']):
                            filtered_epochs.append(curves['epochs'][i])
                
                processed_curves[key] = filtered_values
                processed_curves[f'{key}_epochs'] = filtered_epochs
        
        # Add metadata
        processed_curves['_metadata'] = {
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
            'best_metrics': data.get('best_metrics', {}),
            'total_epochs': len(set(curves['epochs'])),
            'total_steps': max(curves['steps']) if curves['steps'] else 0,
            'additional_metric_names': list(additional_metric_names),
            'config': data.get('config', {}),
            'summary_stats': data.get('summary_stats', {})
        }
        
        return processed_curves
    
    def load_evaluation_report(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a classification evaluation report (sklearn-style) from JSON.
        Expects keys: per-class precision/recall/f1-score and optional 'accuracy'.
        """
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Evaluation report not found: {p}")
        with open(p) as f:
            report = json.load(f)
        return report
    
    def save_plot(self, path: Path, dpi: int = 300, bbox_inches: str = 'tight', 
                  facecolor: str = 'white'):
        """Save the current plot with consistent settings."""
        self._create_save_path(path)
        plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
        plt.close()
        print(f"Saved plot to {path}")
        
    