import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from .common_plots import CommonPlots


class ClassificationPlots(CommonPlots):
    """Plotting utilities specifically for classification tasks."""

    def __init__(
        self,
        training_history_path: Optional[Union[str, Path]] = None,
        eval_history_path: Optional[Union[str, Path]] = None,
        output_directory: Optional[Union[str, Path]] = None
    ):
        super().__init__(output_directory)
        self.history = (
            self.load_training_history(training_history_path)
            if training_history_path
            else None
        )
        self.eval_report = (
            self.load_evaluation_report(eval_history_path)
            if eval_history_path
            else None
        )
        # CACHE TRAINING HISTORY AND EVAL REPORTS FOR REUSE ACROSS PLOTS.

    def plot_training_history(
        self,
        filename: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        include_additional_metrics: bool = True
    ) -> None:
        """Plot loss, accuracy, and up to 4 additional metrics using self.history."""
        if self.history is None:
            print("No training history loaded. Cannot plot training curves.")
            return
            
        curves = self.history
        if not curves.get('train_loss'):
            print("No training loss data found in history.")
            return
            
        additional = curves.get('_metadata', {}).get('additional_metric_names', [])
        num_plots = 2 + (min(len(additional), 4) if include_additional_metrics else 0)

        # Create subplots
        if num_plots <= 2:
            fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
            axes = [axes] if num_plots == 1 else list(axes)
        else:
            rows = (num_plots + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(14, 6 * rows))
            axes = axes.flatten()

        epochs = range(1, len(curves['train_loss']) + 1)
        idx = 0

        # Loss plot
        axes[idx].plot(epochs, curves['train_loss'], 'b-o', label='Train Loss')
        if curves.get('val_loss'):
            axes[idx].plot(epochs, curves['val_loss'], 'r-s', label='Val Loss')
        axes[idx].set_title("Loss")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        idx += 1

        # Accuracy plot
        if curves.get('train_acc') and curves.get('val_acc'):
            tacc = curves['train_acc']
            if tacc and max(tacc) > 2.0:
                tacc = [v / 100.0 for v in tacc]
            axes[idx].plot(epochs, tacc, 'g-o', label='Train Acc')

            vacc = curves['val_acc']
            if vacc and max(vacc) > 2.0:
                vacc = [v / 100.0 for v in vacc]
            axes[idx].plot(epochs, vacc, 'm-s', label='Val Acc')

            axes[idx].set_title("Accuracy")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Accuracy")
            axes[idx].set_ylim(0, 1)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            idx += 1

        # Additional metrics
        colors = ['orange', 'purple', 'brown', 'pink']
        for i, name in enumerate(additional[:4]):
            if idx >= len(axes):
                break
            vals = curves.get(name, [])
            if not vals:
                continue
            ax = axes[idx]
            ax.plot(epochs, vals, color=colors[i], marker='o', label=name.replace('_', ' ').title())
            ax.set_title(name.replace('_', ' ').title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            if any(k in name for k in ['acc', 'f1', 'precision', 'recall']):
                ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.3)
            idx += 1

        # Hide unused subplots
        for j in range(idx, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "classification_curves.png", save_path=save_path)

    def plot_confusion_matrix(
        self,
        filename: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        normalize: bool = False
    ) -> None:
        """Plot confusion matrix loaded from self.eval_report."""
        if self.eval_report is None:
            print("No evaluation report loaded. Cannot plot confusion matrix.")
            return
            
        history_list = self.eval_report.get("metrics_history", [])
        if not history_list:
            print("No metrics history found in evaluation report.")
            return
            
        entry = history_list[0]  # or -1 for latest

        cm = np.array(entry.get("confusion_matrix", []))
        if cm.size == 0:
            print("No confusion matrix data found.")
            return
            
        class_names = entry.get("class_names", [])

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots(
            figsize=(max(8, len(class_names)), max(6, len(class_names)))
        )
        # DRAW HEATMAP OF TRUE VS PREDICTED COUNTS OR PROPORTIONS.
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "confusion_matrix.png", save_path=save_path)

    def plot_classification_report(
        self,
        filename: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Bar chart of precision, recall, and F1 from self.eval_report."""
        if self.eval_report is None:
            print("No evaluation report loaded. Cannot plot classification report.")
            return
            
        history_list = self.eval_report.get("metrics_history", [])
        if not history_list:
            print("No metrics history found in evaluation report.")
            return
            
        entry = history_list[0]  # or -1 for latest

        report = entry.get("classification_report", {})
        if not report:
            print("No classification report data found.")
            return
            
        class_names = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
        if not class_names:
            print("No class-specific metrics found in classification report.")
            return
            
        precision = [report[c]["precision"] for c in class_names]
        recall = [report[c]["recall"] for c in class_names]
        f1 = [report[c]["f1-score"] for c in class_names]

        x = np.arange(len(class_names))
        w = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.5), 6))
        # GROUPED BARS HIGHLIGHT CLASS-WISE PRECISION, RECALL, AND F1.
        ax.bar(x - w, precision, w, label="Precision")
        ax.bar(x, recall, w, label="Recall")
        ax.bar(x + w, f1, w, label="F1-Score")

        ax.set_title("Classification Report")
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "classification_report.png", save_path=save_path)

    def plot_learning_rate_schedule(
        self,
        filename: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot LR schedule from self.history."""
        if self.history is None:
            print("No training history loaded. Cannot plot learning rate schedule.")
            return
            
        lr = self.history.get("learning_rate", [])
        if not lr:
            print("No learning_rate data to plot")
            return

        epochs = range(1, len(lr) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, lr, "b-o")
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "lr_schedule.png", save_path=save_path)
