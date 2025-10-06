import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List
from .common_plots import CommonPlots

class ClassificationPlots(CommonPlots):
    """Plotting utilities specifically for classification tasks."""
    def __init__(self, training_history_path: str = None, eval_history_path: str = None, output_directory: str = None):
        super().__init__(output_directory)
        if training_history_path:
            self.history = self.load_training_history(training_history_path)
        else:
            self.history = None
            
        if eval_history_path:
            self.eval_report = self.load_evaluation_report(eval_history_path)
        else:
            self.eval_report = None

    def plot_training_history(
        self,
        save_path: Optional[str] = None,
        include_additional_metrics: bool = True
    ) -> None:
        """Plot loss, accuracy, and up to 4 additional metrics using self.history."""
        curves = self.history
        path = self._resolve_path(save_path, "training_curves.png")

        additional = curves['_metadata'].get('additional_metric_names', [])
        num_plots = 2 + (min(len(additional), 4) if include_additional_metrics else 0)

        # create subplots
        if num_plots <= 2:
            fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 6))
            axes = ([axes] if num_plots == 1 else list(axes))
        else:
            rows = (num_plots + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(14, 6*rows))
            axes = axes.flatten()

        idx = 0
        # Loss
        if curves.get('train_loss'):
            epochs = range(1, len(curves['train_loss'])+1)
            axes[idx].plot(epochs, curves['train_loss'], 'b-o', label='Train Loss')
            if curves.get('val_loss'):
                axes[idx].plot(epochs, curves['val_loss'], 'r-s', label='Val Loss')
            axes[idx].set(title="Loss", xlabel="Epoch", ylabel="Loss")
            axes[idx].legend(); axes[idx].grid(alpha=0.3)
            idx += 1

        # Accuracy
        if curves.get('train_acc'):
            tacc = curves['train_acc']
            if max(tacc) > 2.0: tacc = [v/100.0 for v in tacc]
            axes[idx].plot(epochs, tacc, 'g-o', label='Train Acc')
            if curves.get('val_acc'):
                vacc = curves['val_acc']
                if max(vacc) > 2.0: vacc = [v/100.0 for v in vacc]
                axes[idx].plot(epochs, vacc, 'm-s', label='Val Acc')
            axes[idx].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0,1))
            axes[idx].legend(); axes[idx].grid(alpha=0.3)
            idx += 1

        # Additional
        colors = ['orange','purple','brown','pink']
        for i, name in enumerate(additional[:4]):
            if not include_additional_metrics or idx >= len(axes): break
            if curves.get(name):
                vals = curves[name]
                ax = axes[idx]
                ax.plot(epochs, vals, color=colors[i], marker='o', label=name.replace('_',' ').title())
                ax.set(title=name.replace('_',' ').title(), xlabel="Epoch", ylabel=name)
                if any(k in name for k in ['acc','f1','precision','recall']): ax.set_ylim(0,1)
                ax.legend(); ax.grid(alpha=0.3)
                idx += 1

        # hide unused
        for j in range(idx, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        self.save_plot(path)
    
    
    def plot_confusion_matrix(
            self,
            save_path: Optional[str] = None,
            normalize: bool = False
        ) -> None:
        """
        Plot confusion matrix loaded from self.eval_report['confusion_matrix'].
        """
        # Get the list of history entries
        history_list = self.eval_report.get("metrics_history", [])

        # Select the entry you want (e.g., first or last)
        entry = history_list[0]  # or [-1]

        # Extract confusion matrix and class names
        cm = np.array(entry.get("confusion_matrix", []))
        class_names = entry.get("class_names", [])
        path = self._resolve_path(save_path, "confusion_matrix.png")

        fmt = ".2f" if normalize else "d"
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names))))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"label": "Proportion" if normalize else "Count"})
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        self.save_plot(path)
    
    def plot_classification_report(self, save_path=None):
        history_list = self.eval_report.get("metrics_history", [])

        # Select the entry you want (e.g., first or last)
        entry = history_list[0]  # or [-1]

        report = entry.get("classification_report", {})
        class_names = [k for k in report if k not in ("accuracy","macro avg","weighted avg")]
        precision = [report[c]["precision"] for c in class_names]
        recall    = [report[c]["recall"]    for c in class_names]
        f1        = [report[c]["f1-score"]  for c in class_names]

        x = np.arange(len(class_names)); w=0.25
        fig, ax = plt.subplots(figsize=(max(10,len(class_names)*1.5),6))
        ax.bar(x-w, precision, w, label="Precision")
        ax.bar(x,   recall,    w, label="Recall")
        ax.bar(x+w, f1,        w, label="F1-Score")
        ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set(title="Classification Report", ylim=(0,1), xlabel="Class", ylabel="Score")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = self._resolve_path(save_path, "classification_report.png")
        self.save_plot(path)

    
    def plot_learning_rate_schedule(
    self,
    save_path: Optional[str] = None,
    ) -> None:
        """Plot LR schedule from self.history only once loaded."""
        curves = self.history
        if not curves.get("learning_rate"):
            print("No learning_rate data to plot"); return
        lr = curves["learning_rate"]
        epochs = range(1, len(lr)+1)
        path = self._resolve_path(save_path, "lr_schedule.png")
        plt.figure(figsize=(10,6))
        plt.plot(epochs, lr, "b-o"); plt.xscale("linear"); plt.yscale("log")
        plt.title("Learning Rate Schedule"); plt.xlabel("Epoch"); plt.ylabel("LR")
        plt.grid(alpha=0.3); plt.tight_layout()
        self.save_plot(path)
        