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
        """Plot loss, accuracy, gradients, and learning rate trends."""
        if self.history is None:
            print("No training history loaded. Cannot plot training curves.")
            return

        curves = self.history
        if not curves.get("train_loss"):
            print("No training loss data found in history.")
            return

        fallback_epochs = list(curves.get("train_loss_epochs") or curves.get("epochs", []))
        if not fallback_epochs:
            fallback_epochs = list(range(1, len(curves["train_loss"]) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.subplots_adjust(hspace=0.32, wspace=0.28)
        ax_loss, ax_acc, ax_grad, ax_lr = axes.flatten()

        # ---- Loss panel -----------------------------------------------------
        train_loss_epochs, train_loss = self.extract_series(curves, "train_loss", fallback_epochs)
        val_loss_epochs, val_loss = self.extract_series(curves, "val_loss", fallback_epochs)
        if train_loss is not None:
            ax_loss.plot(train_loss_epochs, train_loss, color=self.palette[0], linewidth=2.2, label="Train")
        if val_loss is not None and np.isfinite(val_loss).any():
            ax_loss.plot(val_loss_epochs, val_loss, color=self.palette[3], linewidth=2.0, linestyle="--", label="Val")
            best_idx = int(np.nanargmin(val_loss))
            ax_loss.scatter(val_loss_epochs[best_idx], val_loss[best_idx], color=self.palette[3], edgecolors="white", s=60, zorder=5)
        self._style_axis(ax_loss, title="Cross-Entropy Loss", xlabel="Epoch", ylabel="Loss", grid=False, facecolor='white')
        if ax_loss.has_data():
            ax_loss.legend(frameon=False)

        # ---- Accuracy panel -------------------------------------------------
        train_acc_epochs, train_acc = self.extract_series(curves, "train_acc", fallback_epochs)
        val_acc_epochs, val_acc = self.extract_series(curves, "val_acc", fallback_epochs)
        if train_acc is not None or val_acc is not None:
            if train_acc is not None:
                ax_acc.plot(train_acc_epochs, train_acc, color=self.palette[1], linewidth=2.2, label="Train")
            if val_acc is not None:
                ax_acc.plot(val_acc_epochs, val_acc, color=self.palette[2], linewidth=2.0, linestyle="--", label="Val")
            ax_acc.set_ylim(0, 1.02)
            self._style_axis(ax_acc, title="Accuracy", xlabel="Epoch", ylabel="Accuracy", grid=False, facecolor='white')
            ax_acc.legend(frameon=False)
        else:
            ax_acc.axis("off")

        # ---- Gradient panel -------------------------------------------------
        grad_epochs, grad_avg = self.extract_series(curves, "grad_norm_avg", fallback_epochs)
        _, grad_max = self.extract_series(curves, "grad_norm_max", fallback_epochs)
        clip_epochs, grad_clip = self.extract_series(curves, "grad_clip_frac", fallback_epochs)

        if grad_avg is not None or grad_max is not None or grad_clip is not None:
            if grad_avg is not None:
                ax_grad.plot(grad_epochs, grad_avg, color=self.palette[4], linewidth=2.0, label="Grad norm (avg)")
            if grad_max is not None:
                ax_grad.plot(grad_epochs, grad_max, color=self.palette[5], linewidth=1.8, linestyle="--", label="Grad norm (max)")
            if grad_clip is not None:
                clip_percent = grad_clip * 100.0
                ax_clip = ax_grad.twinx()
                ax_clip.bar(clip_epochs, clip_percent, width=0.4, alpha=0.25, color=self.palette[6])
                ax_clip.set_ylabel("Clip %", color=self.palette[6])
                ax_clip.set_ylim(0, max(clip_percent) * 1.2 if clip_percent.size else 1)
                ax_clip.tick_params(axis='y', labelcolor=self.palette[6])
            self._style_axis(ax_grad, title="Gradient behaviour", xlabel="Epoch", ylabel="Norm", grid=False, facecolor='white')
            ax_grad.legend(frameon=False, loc="upper right")
        else:
            ax_grad.axis("off")

        # ---- Learning rate panel -------------------------------------------
        lr_epochs, lr_vals = self.extract_series(curves, "learning_rate", fallback_epochs)
        plotted_values = []
        if lr_vals is not None and np.all(lr_vals > 0):
            ax_lr.plot(lr_epochs, lr_vals, color=self.palette[7], linewidth=2.0, label="Base")
            plotted_values.append(lr_vals)
        for key in sorted(k for k in curves.keys() if k.startswith("lr_group_") and not k.endswith("_epochs")):
            group_epochs, group_vals = self.extract_series(curves, key, fallback_epochs)
            if group_vals is not None and np.all(group_vals > 0):
                label = key.replace("_", " ").title()
                ax_lr.plot(group_epochs, group_vals, linewidth=1.6, linestyle="--", label=label)
                plotted_values.append(group_vals)
        if plotted_values:
            min_lr = min(float(np.nanmin(vals)) for vals in plotted_values if np.all(vals > 0))
            if min_lr > 0:
                ax_lr.set_yscale('log')
            self._style_axis(ax_lr, title="Learning rate", xlabel="Epoch", ylabel="LR", grid=False, facecolor='white')
            ax_lr.legend(frameon=False)
        else:
            ax_lr.axis("off")

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
