import json
import numpy as np
import matplotlib

# Force a non-interactive backend early so matplotlib never spins up GUI helper threads.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Any, Dict, Optional, Union

class CommonPlots:
    """Base class for plotting, with unified save logic."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        # SET DEFAULT OUTPUT LOCATION AND GLOBAL MATPLOTLIB STYLE.
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        plt.ioff()
        self._create_save_path(self.output_dir)
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Set up consistent plotting style across all plots."""
        sns.set_theme(context="talk", style="whitegrid")
        # Default palette prioritises good contrast on light backgrounds.
        palette = sns.color_palette("deep")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.titlesize': 14,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 12,
            'axes.facecolor': '#f7f9fc',
            'figure.facecolor': 'white',
            'font.size': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.edgecolor': '#d6d9de',
            'axes.linewidth': 0.8,
            'grid.alpha': 0.35,
            'grid.linestyle': '--',
            'savefig.bbox': 'tight',
        })
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
        self._axis_facecolor = '#f7f9fc'
    
    def _create_save_path(self, path: Path) -> Path:
        """Create save_path directory if it doesn't exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent

    def _style_axis(
        self,
        ax: plt.Axes,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: bool = True,
        facecolor: Optional[str] = None,
    ) -> None:
        """Apply a consistent visual treatment to axes."""
        if facecolor is None:
            facecolor = getattr(self, "_axis_facecolor", None)
        if facecolor:
            ax.set_facecolor(facecolor)
        if title is not None:
            ax.set_title(title, fontsize=14, fontweight='semibold', pad=12)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        if grid:
            ax.grid(True, which="major", color="#d0d6e0", linewidth=0.8, alpha=0.5)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)
        ax.tick_params(axis='both', colors='#3d3d3d')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#d6d9de')
        ax.spines['bottom'].set_color('#d6d9de')
    
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
    
    def save_plot(
        self,
        fig: plt.Figure,
        filename: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        bbox_inches: str = 'tight',
        facecolor: str = 'white'
    ) -> Path:
        """
        Save a matplotlib Figure with minimal ceremony.

        Args:
            fig: the Figure to save
            filename: desired filename (with extension); default 'plot.png'
            save_dir: directory path to save into; defaults to instance output_dir
            dpi, bbox_inches, facecolor: passed to plt.savefig

        Returns:
            The full Path to the saved file.
        """
        # Determine directory
        directory = Path(save_path) if save_path else self.output_dir
        directory.mkdir(parents=True, exist_ok=True)

        # Determine filename
        fname = filename or "plot.png"
        path = directory / fname

        # Save and close
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
        plt.close(fig)
        print(f"Saved plot -> {path}")
        return path
        
    
