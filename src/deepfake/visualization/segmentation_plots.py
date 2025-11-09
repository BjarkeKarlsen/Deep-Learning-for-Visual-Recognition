from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import seaborn as sns
from matplotlib import ticker as mticker
from matplotlib.patches import Patch

from deepfake.visualization.common_plots import CommonPlots


class SegmentationPlots(CommonPlots):
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
        # STORE OPTIONAL HISTORY AND EVAL REPORT REFERENCES FOR LATER PLOTS.

        self.colors = {
            "train_loss": "#1f77b4",
            "val_loss": "#d62728",
            "train_dice": "#2ca02c",
            "val_dice": "#9467bd",
            "bce_loss": "#17becf",
            "focal_loss": "#17becf",
            "dice_loss": "#ffbb78",
            "learning_rate": "#ff7f0e",
            "bucket_bar": "#4c72b0",
            "bucket_count": "#dd8452",
            "threshold_precision": "#1f77b4",
            "threshold_recall": "#2ca02c",
            "threshold_dice": "#9467bd",
            "background_cov": "#4c72b0",
            "background_max": "#dd8452",
            "grad_norm": "#e377c2",
            "grad_clip": "#7f7f7f",
        }

    def plot_segmentation(self, image: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray, 
                         filename: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plot the input image, the ground-truth mask, and a predicted-mask overlay.

        Args:
            image (H×W×C or H×W numpy): Input image.
            true_mask (H×W numpy): Ground-truth segmentation (0/1).
            pred_mask (H×W numpy): Predicted segmentation (0/1 or probabilities).
            filename: Optional filename for the saved plot.
            save_path: Optional directory path to save the figure.
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('white')

        prepared_img = self._prepare_image(image)
        true_overlay = self.overlay_mask(prepared_img, true_mask, color=(0, 1, 0))
        pred_overlay = self.overlay_mask(prepared_img, pred_mask, color=(1, 0, 0))
        dice_score = self._compute_dice_score(true_mask, pred_mask)
        error_overlay, legend_handles = self.compute_error_overlay(prepared_img, true_mask, pred_mask)

        axes[0].imshow(prepared_img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(true_overlay)
        axes[1].set_title('Ground Truth Overlay', fontsize=13, pad=10)
        axes[1].axis('off')

        axes[2].imshow(pred_overlay)
        axes[2].set_title(f'Prediction Overlay\nDice {dice_score:.3f}', fontsize=13, pad=10)
        axes[2].axis('off')

        axes[3].imshow(error_overlay)
        axes[3].set_title('Error Map', fontsize=13, pad=10)
        axes[3].axis('off')
        if legend_handles:
            axes[3].legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=len(legend_handles),
                frameon=False,
                fontsize=10,
            )

        plt.tight_layout()
        
        # Use the base class save_plot method
        self.save_plot(fig, filename=filename or "segmentation_plot.png", save_path=save_path)

    def plot_segmentation_gallery(self,
                                 examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 ncols: int = 3,
                                 filename: Optional[str] = None,
                                 save_path: Optional[str] = None,
                                 title_prefix: str = "Example",
                                 example_scale: float = 1.35) -> None:
        """
        Plot a grid of segmentation examples.
        
        Args:
            examples: list of (image, true_mask, pred_mask) tuples
            ncols: number of columns per row (each example uses 3 subplots)
            filename: Optional filename for the saved plot.
            save_path: Optional directory path to save the figure.
        """
        if not examples:
            return

        n = len(examples)
        nrows = int(np.ceil(n / ncols))
        show_error_map = True
        panels_per_example = 4 if show_error_map else 3
        base_width = 2.2 * example_scale
        base_height = 2.6 * example_scale
        fig_width = min(32.0, max(12.0, ncols * panels_per_example * base_width))
        fig_height = min(18.0, max(5.0, nrows * base_height))
        fig, axes = plt.subplots(nrows, ncols * panels_per_example, figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('white')
        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)

        legend_added = False

        for idx, (img, true_m, pred_m) in enumerate(examples):
            row = idx // ncols
            col_offset = (idx % ncols) * panels_per_example

            prepared_img = self._prepare_image(img)
            true_overlay = self.overlay_mask(prepared_img, true_m, color=(0, 1, 0))
            pred_overlay = self.overlay_mask(prepared_img, pred_m, color=(1, 0, 0))
            dice_score = self._compute_dice_score(true_m, pred_m)

            ax_image = axes[row, col_offset]
            ax_image.imshow(prepared_img)
            ax_image.set_title(f"{title_prefix} #{idx+1} • Input", fontsize=13, pad=10)
            ax_image.axis("off")

            ax_true = axes[row, col_offset + 1]
            ax_true.imshow(true_overlay)
            ax_true.set_title("Ground Truth", fontsize=12, pad=10)
            ax_true.axis("off")

            ax_pred = axes[row, col_offset + 2]
            ax_pred.imshow(pred_overlay)
            ax_pred.set_title(f"Prediction (Dice {dice_score:.3f})", fontsize=12, pad=10)
            ax_pred.axis("off")

            if show_error_map:
                error_overlay, legend_handles = self.compute_error_overlay(prepared_img, true_m, pred_m)
                ax_err = axes[row, col_offset + 3]
                ax_err.imshow(error_overlay)
                ax_err.set_title("Error Map", fontsize=12, pad=10)
                ax_err.axis("off")
                if not legend_added and legend_handles:
                    ax_err.legend(
                        handles=legend_handles,
                        loc="lower center",
                        bbox_to_anchor=(0.5, -0.25),
                        ncol=len(legend_handles),
                        frameon=False,
                        fontsize=10,
                    )
                    legend_added = True

        # HIDE ANY UNUSED AXES SO EMPTY PANELS DO NOT SHOW UP.
        total_examples = nrows * ncols
        for unused_idx in range(n, total_examples):
            row = unused_idx // ncols
            col_offset = (unused_idx % ncols) * panels_per_example
            for panel in range(panels_per_example):
                axes[row, col_offset + panel].axis("off")

        plt.tight_layout()
        
        # USE THE BASE CLASS SAVE_PLOT METHOD TO HANDLE FILE OUTPUT.
        self.save_plot(fig, filename=filename or "segmentation_gallery.png", save_path=save_path)

    def plot_training_history(self, filename: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """Plot segmentation training diagnostics with loss, dice, components, and gradients."""
        if self.history is None:
            print("No training history loaded. Cannot plot training curves.")
            return

        curves = self.history
        train_loss_series = curves.get("train_loss", [])
        if not train_loss_series:
            print("No training loss data found in history.")
            return

        fallback_epochs = list(curves.get("train_loss_epochs") or curves.get("epochs", []))
        if not fallback_epochs:
            fallback_epochs = list(range(1, len(train_loss_series) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        fig.subplots_adjust(hspace=0.32, wspace=0.28)
        ax_loss, ax_dice, ax_components, ax_grad = axes.flatten()

        # Loss panel
        train_loss_epochs, train_loss_vals = self.extract_series(curves, "train_loss", fallback_epochs)
        val_loss_epochs, val_loss_vals = self.extract_series(curves, "val_loss", fallback_epochs)
        if train_loss_vals is not None:
            ax_loss.plot(train_loss_epochs, train_loss_vals, color=self.colors["train_loss"], linewidth=2.2, label="Train")
        if val_loss_vals is not None and np.isfinite(val_loss_vals).any():
            ax_loss.plot(val_loss_epochs, val_loss_vals, color=self.colors["val_loss"], linewidth=2.0, linestyle="--", label="Val")
            best_idx = int(np.nanargmin(val_loss_vals))
            ax_loss.scatter(val_loss_epochs[best_idx], val_loss_vals[best_idx], color=self.colors["val_loss"], edgecolors="white", s=70, zorder=5)
        self._style_axis(ax_loss, title="Loss", xlabel="Epoch", ylabel="Loss", grid=False, facecolor='white')
        if ax_loss.has_data():
            ax_loss.legend(frameon=False, loc="upper right")

        # Dice panel
        train_dice_epochs, train_dice_vals = self.extract_series(curves, "train_dice", fallback_epochs)
        val_dice_epochs, val_dice_vals = self.extract_series(curves, "val_dice", fallback_epochs)
        if train_dice_vals is not None:
            ax_dice.plot(train_dice_epochs, train_dice_vals, color=self.colors["train_dice"], linewidth=2.2, label="Train")
        if val_dice_vals is not None and np.isfinite(val_dice_vals).any():
            ax_dice.plot(val_dice_epochs, val_dice_vals, color=self.colors["val_dice"], linewidth=2.0, linestyle="--", label="Val")
            best_idx = int(np.nanargmax(val_dice_vals))
            ax_dice.scatter(val_dice_epochs[best_idx], val_dice_vals[best_idx], color=self.colors["val_dice"], edgecolors="white", s=70, zorder=5)
        self._style_axis(ax_dice, title="Dice score", xlabel="Epoch", ylabel="Dice", grid=False, facecolor='white')
        ax_dice.set_ylim(0, 1.02)
        if ax_dice.has_data():
            ax_dice.legend(frameon=False, loc="lower right")

        # Loss component panel
        primary_name = "focal" if "train_focal_weighted" in curves or "val_focal_weighted" in curves else "bce"
        comp_color = self.colors.get(f"{primary_name}_loss", self.colors["bce_loss"])
        component_keys = [
            (f"train_{primary_name}_weighted", "Train primary", comp_color, "-"),
            (f"val_{primary_name}_weighted", "Val primary", comp_color, "--"),
            ("train_dice_weighted", "Train dice", self.colors["dice_loss"], "-"),
            ("val_dice_weighted", "Val dice", self.colors["dice_loss"], "--"),
        ]
        plotted_components = False
        for key, label, color, style in component_keys:
            epochs_arr, values_arr = self.extract_series(curves, key, fallback_epochs)
            if values_arr is not None:
                ax_components.plot(epochs_arr, values_arr, color=color, linewidth=2.0, linestyle=style, label=label)
                plotted_components = True
        if plotted_components:
            self._style_axis(ax_components, title="Weighted loss components", xlabel="Epoch", ylabel="Loss", grid=False, facecolor='white')
            ax_components.legend(frameon=False, loc="upper right")
        else:
            ax_components.axis("off")

        # Gradients & learning rate panel
        grad_epochs, grad_avg = self.extract_series(curves, "grad_norm_avg", fallback_epochs)
        _, grad_max = self.extract_series(curves, "grad_norm_max", fallback_epochs)
        clip_epochs, clip_vals = self.extract_series(curves, "grad_clip_frac", fallback_epochs)
        lr_epochs, lr_vals = self.extract_series(curves, "learning_rate", fallback_epochs)

        ax_grad_plotted = False
        if grad_avg is not None:
            ax_grad.plot(grad_epochs, grad_avg, color=self.colors["grad_norm"], linewidth=2.0, label="Grad norm (avg)")
            ax_grad_plotted = True
        if grad_max is not None:
            ax_grad.plot(grad_epochs, grad_max, color=self.colors["grad_norm"], linewidth=1.8, linestyle="--", alpha=0.75, label="Grad norm (max)")
            ax_grad_plotted = True

        clip_axis = None
        if clip_vals is not None:
            clip_percent = clip_vals * 100.0
            clip_axis = ax_grad.twinx()
            clip_axis.bar(clip_epochs, clip_percent, width=0.4, alpha=0.25, color=self.colors["grad_clip"], label="Clip %")
            clip_axis.set_ylabel("Clip %", color=self.colors["grad_clip"])
            clip_axis.tick_params(axis='y', labelcolor=self.colors["grad_clip"])
            clip_axis.set_ylim(0, max(clip_percent) * 1.2 if clip_percent.size else 1)

        lr_series = []
        if lr_vals is not None and np.all(lr_vals > 0):
            lr_series.append((lr_epochs, lr_vals, "LR (base)"))
        for key in sorted(k for k in curves if k.startswith("lr_group_") and not k.endswith("_epochs")):
            g_epochs, g_vals = self.extract_series(curves, key, fallback_epochs)
            if g_vals is not None and np.all(g_vals > 0):
                lr_series.append((g_epochs, g_vals, key.replace("_", " ").title()))

        lr_axis = None
        if lr_series:
            lr_axis = clip_axis if clip_axis is not None else ax_grad.twinx()
            for idx, (epochs_arr, values_arr, label) in enumerate(lr_series):
                lr_axis.plot(epochs_arr, values_arr, linewidth=1.8, linestyle="-" if idx == 0 else "--", color=self.colors["learning_rate"], alpha=0.85 if idx == 0 else 0.65, label=label)
            min_lr = min(float(np.nanmin(values_arr)) for _, values_arr, _ in lr_series)
            if min_lr > 0:
                lr_axis.set_yscale('log')
            lr_axis.set_ylabel("Learning rate", color=self.colors["learning_rate"])
            lr_axis.tick_params(axis='y', labelcolor=self.colors["learning_rate"])

        if ax_grad_plotted or clip_axis is not None or lr_axis is not None:
            self._style_axis(ax_grad, title="Gradients & learning rate", xlabel="Epoch", ylabel="Norm", grid=False, facecolor='white')
            handles, labels = ax_grad.get_legend_handles_labels()
            if clip_axis is not None:
                h_clip, l_clip = clip_axis.get_legend_handles_labels()
                handles.extend(h_clip)
                labels.extend(l_clip)
            if lr_axis is not None:
                h_lr, l_lr = lr_axis.get_legend_handles_labels()
                handles.extend(h_lr)
                labels.extend(l_lr)
            if handles:
                ax_grad.legend(handles, labels, frameon=False, loc="upper right")
        else:
            ax_grad.axis("off")

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "training_history.png", save_path=save_path)

    def plot_learning_rate_schedule(self, filename: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """Plot LR schedule from self.history."""
        if self.history is None:
            print("No training history loaded. Cannot plot learning rate schedule.")
            return
            
        curves = self.history
        lr_curve = curves.get("learning_rate")
        group_keys = [
            key for key in curves.keys()
            if key.startswith("lr_group_") and not key.endswith("_epochs")
        ]

        if not lr_curve and not group_keys:
            print("No learning_rate data to plot")
            return

        lr_values = []
        if lr_curve:
            lr_values = [value if value is not None else np.nan for value in lr_curve]

        epochs = np.arange(1, len(lr_values) + 1) if lr_values else None
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor('white')

        if lr_values:
            lr_array = np.asarray(lr_values, dtype=float)
            ax.plot(
                epochs,
                lr_array,
                color=self.colors["learning_rate"],
                marker='o',
                markerfacecolor='white',
                markeredgecolor=self.colors["learning_rate"],
                linewidth=2.2,
                markersize=5,
                label='Aggregate LR',
            )
            ax.scatter(
                epochs[-1],
                lr_array[-1],
                s=90,
                color=self.colors["learning_rate"],
                edgecolors='white',
                linewidth=1.4,
                zorder=6,
            )

        if group_keys:
            subgroup_palette = sns.color_palette("rocket", len(group_keys))
            for idx, key in enumerate(sorted(group_keys)):
                values = curves.get(key, [])
                if not values:
                    continue
                epochs_key = curves.get(f"{key}_epochs") or list(range(1, len(values) + 1))
                epochs_arr = np.asarray(epochs_key, dtype=float)
                values_arr = np.asarray(values, dtype=float)
                ax.plot(
                    epochs_arr,
                    values_arr,
                    color=subgroup_palette[idx],
                    linestyle='--',
                    marker='o',
                    markerfacecolor='white',
                    markeredgecolor=subgroup_palette[idx],
                    linewidth=1.8,
                    markersize=4,
                    label=key,
                )

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1e}"))
        self._style_axis(ax, title="Learning Rate Schedule", xlabel="Epoch", ylabel="Learning Rate")
        ax.set_xlim(left=1)
        if group_keys or lr_values:
            ax.legend(loc="best", frameon=False)

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "learning_rate_schedule.png", save_path=save_path)

    def plot_bucket_metrics(self, bucket_metrics: Dict[str, Dict[str, float]],
                            filename: Optional[str] = None,
                            save_path: Optional[str] = None) -> None:
        if not bucket_metrics:
            return
        labels = list(bucket_metrics.keys())
        mean_dice = np.array([bucket_metrics[label].get("mean_dice", np.nan) for label in labels], dtype=float)
        counts = np.array([bucket_metrics[label].get("count", 0) for label in labels], dtype=float)
        x = np.arange(len(labels))
        max_count = counts.max() if counts.size else 0.0

        fig, ax1 = plt.subplots(figsize=(11.5, 6))
        fig.patch.set_facecolor('white')

        bar_colors = sns.color_palette("light:steelblue", len(labels))
        bars = ax1.bar(
            x,
            mean_dice,
            color=bar_colors,
            edgecolor="#4c566a",
            linewidth=0.6,
            width=0.55,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        self._style_axis(
            ax1,
            title="Dice by Tampered Area Bucket",
            xlabel="Tampered Area (% of image pixels)",
            ylabel="Mean Dice",
        )
        ax1.set_ylim(0, 1.05)
        ax1.margins(x=0.05)

        for bar, dice in zip(bars, mean_dice):
            if np.isnan(dice):
                continue
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                dice + 0.03,
                f"{dice:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color="#30455c",
            )

        ax2 = ax1.twinx()
        ax2.set_ylim(0, max(max_count * 1.25, 1.0))
        ax2.plot(
            x,
            counts,
            color=self.colors["bucket_count"],
            marker='o',
            markerfacecolor='white',
            markeredgecolor=self.colors["bucket_count"],
            linewidth=2.0,
            markersize=5,
            label="Sample Count",
        )
        ax2.fill_between(x, counts, color=self.colors["bucket_count"], alpha=0.10)
        self._style_axis(ax2, ylabel="Sample Count", grid=False, facecolor='none')
        for xi, count in zip(x, counts):
            ax2.text(
                xi,
                count + (max_count * 0.05 if max_count > 0 else 0.5),
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=self.colors["bucket_count"],
            )
        ax2.legend(loc="upper right", frameon=False)

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "bucket_metrics.png", save_path=save_path)

    def plot_threshold_sweep(self, threshold_metrics: Dict[str, Dict[str, float]],
                             filename: Optional[str] = None,
                             save_path: Optional[str] = None) -> None:
        if not threshold_metrics:
            return
        threshold_keys = sorted(threshold_metrics.keys(), key=lambda x: float(x))
        thresholds = np.array([float(t) for t in threshold_keys], dtype=float)
        dice_vals = np.array([threshold_metrics[t].get("dice", np.nan) for t in threshold_keys], dtype=float)
        precision_vals = np.array([threshold_metrics[t].get("precision", np.nan) for t in threshold_keys], dtype=float)
        recall_vals = np.array([threshold_metrics[t].get("recall", np.nan) for t in threshold_keys], dtype=float)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor('white')

        ax.plot(
            thresholds,
            dice_vals,
            color=self.colors["threshold_dice"],
            marker='o',
            markerfacecolor='white',
            markeredgecolor=self.colors["threshold_dice"],
            linewidth=2.2,
            markersize=5,
            label="Dice",
        )
        ax.fill_between(thresholds, dice_vals, color=self.colors["threshold_dice"], alpha=0.10)
        ax.plot(
            thresholds,
            precision_vals,
            color=self.colors["threshold_precision"],
            marker='s',
            markerfacecolor='white',
            markeredgecolor=self.colors["threshold_precision"],
            linewidth=2.0,
            markersize=5,
            label="Precision",
        )
        ax.plot(
            thresholds,
            recall_vals,
            color=self.colors["threshold_recall"],
            marker='^',
            markerfacecolor='white',
            markeredgecolor=self.colors["threshold_recall"],
            linewidth=2.0,
            markersize=5,
            label="Recall",
        )

        if np.isfinite(dice_vals).any():
            best_idx = int(np.nanargmax(dice_vals))
            best_threshold = thresholds[best_idx]
            best_dice = dice_vals[best_idx]
            ax.axvline(best_threshold, color="#9aa0ac", linestyle="--", linewidth=1.3, alpha=0.7)
            ax.scatter(
                best_threshold,
                best_dice,
                color=self.colors["threshold_dice"],
                s=110,
                edgecolors='white',
                linewidth=1.4,
                zorder=6,
            )
            ax.annotate(
                f"Suggested threshold: {best_threshold:.2f}\nDice {best_dice:.3f}",
                xy=(best_threshold, best_dice),
                xytext=(best_threshold + 0.05, min(0.98, best_dice + 0.08)),
                arrowprops=dict(arrowstyle="->", color=self.colors["threshold_dice"], lw=1.3),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=self.colors["threshold_dice"], alpha=0.85),
            )

        self._style_axis(ax, title="Threshold Sweep", xlabel="Threshold", ylabel="Score")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
        ax.legend(loc="best", frameon=False)

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "threshold_sweep.png", save_path=save_path)

    def plot_background_histograms(self, background_stats: Dict[str, Any],
                                   filename: Optional[str] = None,
                                   save_path: Optional[str] = None) -> None:
        if not background_stats:
            return

        bins = background_stats.get("background_hist_bins")
        cov_counts = background_stats.get("background_hist_coverage")
        max_counts = background_stats.get("background_hist_max")

        if not bins or cov_counts is None or max_counts is None:
            return

        bins = np.array(bins, dtype=float)
        cov_counts = np.array(cov_counts, dtype=float)
        max_counts = np.array(max_counts, dtype=float)
        centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.42

        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        fig.patch.set_facecolor('white')

        ax.bar(
            centers - width / 2,
            cov_counts,
            width=width,
            label="Mean Probability",
            alpha=0.75,
            color=self.colors["background_cov"],
            edgecolor="#3a5f7d",
            linewidth=0.4,
        )
        ax.bar(
            centers + width / 2,
            max_counts,
            width=width,
            label="Max Probability",
            alpha=0.68,
            color=self.colors["background_max"],
            edgecolor="#8f4b20",
            linewidth=0.4,
        )
        ax.set_xlim(0, 1.0)
        ax.margins(x=0)
        self._style_axis(ax, title="Background False-Positive Distribution", xlabel="Probability", ylabel="Image Count")
        ax.legend(loc="upper right", frameon=False)

        over_half = background_stats.get("background_over_0.5")
        sample_count = background_stats.get("background_samples")
        mean_prob = background_stats.get("background_mean_probability")
        mean_max = background_stats.get("background_mean_max_probability")
        summary_lines = []
        if sample_count:
            if over_half is not None:
                summary_lines.append(f">{0.5:.1f} prob in {over_half}/{sample_count} images")
            if mean_prob is not None and mean_max is not None:
                summary_lines.append(f"Mean prob: {mean_prob:.3f} • Mean max: {mean_max:.3f}")
        if summary_lines:
            ax.text(
                0.02,
                0.95,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#666666", alpha=0.9),
            )

        plt.tight_layout()
        self.save_plot(fig, filename=filename or "background_fp_hist.png", save_path=save_path)

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray,
                    alpha: float = 0.5,
                    color: Tuple[float, float, float] = (1, 0, 0)) -> np.ndarray:
        """
        Return the image with mask overlaid in the given RGB color.
        
        Args:
            image: H×W×3 float [0–1] or uint8 [0–255]
            mask: H×W binary or float probability
            alpha: Transparency level for overlay
            color: RGB color tuple for overlay
            
        Returns:
            Image with mask overlay
        """
        # BLEND MASK INTO THE IMAGE USING THE REQUESTED COLOUR AND ALPHA.
        img = self._prepare_image(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # Build color overlay
        overlay = np.zeros_like(img)
        overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)
        
        return cv2.addWeighted(img, 1.0, overlay, alpha, 0)

    def compute_error_overlay(
        self,
        image: np.ndarray,
        true_mask: np.ndarray,
        pred_mask: np.ndarray,
        alpha: float = 0.6,
    ) -> Tuple[np.ndarray, List[Patch]]:
        """Return overlay highlighting TP/FP/FN regions."""
        img = self._prepare_image(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        true_bin = true_mask > 0.5
        pred_bin = pred_mask > 0.5

        tp = true_bin & pred_bin
        fp = (~true_bin) & pred_bin
        fn = true_bin & (~pred_bin)

        overlay = np.zeros_like(img)
        colors = {
            "TP": (76, 175, 80),       # green
            "FP": (255, 152, 0),       # orange
            "FN": (156, 39, 176),      # purple
        }
        overlay[tp] = colors["TP"]
        overlay[fp] = colors["FP"]
        overlay[fn] = colors["FN"]

        blended = cv2.addWeighted(img, 1.0, overlay, alpha, 0)
        labels = []
        if tp.any():
            labels.append(Patch(color=np.array(colors["TP"]) / 255.0, label="TP"))
        if fp.any():
            labels.append(Patch(color=np.array(colors["FP"]) / 255.0, label="FP"))
        if fn.any():
            labels.append(Patch(color=np.array(colors["FN"]) / 255.0, label="FN"))
        if not labels:
            # Ensure legend still communicates the color code
            labels = [
                Patch(color=np.array(colors["TP"]) / 255.0, label="TP"),
                Patch(color=np.array(colors["FP"]) / 255.0, label="FP"),
                Patch(color=np.array(colors["FN"]) / 255.0, label="FN"),
            ]
        return blended, labels

    def _compute_dice_score(self, true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Compute Dice score between two binary masks."""
        true_bin = (np.asarray(true_mask) > 0.5).astype(np.float32)
        pred_bin = (np.asarray(pred_mask) > 0.5).astype(np.float32)
        intersection = float((true_bin * pred_bin).sum())
        denom = float(true_bin.sum() + pred_bin.sum())
        if denom == 0.0:
            return 1.0 if true_bin.sum() == 0 and pred_bin.sum() == 0 else 0.0
        return (2.0 * intersection + 1e-7) / (denom + 1e-7)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Return an RGB uint8 image for plotting."""
        if image.ndim == 2:
            img = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            img = np.repeat(image, 3, axis=2)
        else:
            img = image.copy()

        if img.dtype != np.uint8:
            max_val = img.max()
            min_val = img.min()
            if max_val <= 1.0 and min_val >= 0.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return img
