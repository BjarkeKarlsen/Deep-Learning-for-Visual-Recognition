from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

    def plot_segmentation(self, image, mask, save_path: Optional[str] = None) -> None:
        """
        Plot the input image, the segmentation mask, and an overlay of the mask on the image.

        Args:
            image (np.ndarray): The input image (H x W x C or H x W).
            mask (np.ndarray): The segmentation mask (H x W), values 0/1 or probabilities.
            save_path (str, optional): If provided, save the plot to this path.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Show image
        axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[0].set_title('Image')
        axes[0].axis('off')

        # Show mask
        axes[1].imshow(mask, cmap='jet', alpha=1.0)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')

        # Overlay mask on image
        axes[2].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[2].imshow(mask, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved segmentation plot to {save_path}")
        else:
            plt.show()
        plt.close()
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot segmentation loss and Dice curves extracted from the history dictionary."""
        path = self._resolve_path(save_path, "segmentation_curves.png")
        curves = self.history

        # Number of epochs based on train_loss length
        n_epochs = len(curves.get('train_loss', []))
        epochs = range(1, n_epochs + 1)

        # ALWAYS grab full-length dice lists, filling missing with NaN
        raw_train_dice = curves.get('train_dice', [])
        raw_val_dice   = curves.get('val_dice', [])
        train_dice = list(raw_train_dice) + [np.nan] * (n_epochs - len(raw_train_dice))
        val_dice   = list(raw_val_dice)   + [np.nan] * (n_epochs - len(raw_val_dice))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Loss plot
        ax1.plot(epochs, curves.get('train_loss', []), 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in curves:
            ax1.plot(epochs, curves.get('val_loss', [np.nan]*n_epochs), 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Segmentation Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Dice plot
        ax2.plot(epochs, train_dice, 'g-', label='Training Dice', linewidth=2)
        ax2.plot(epochs, val_dice,   'm-', label='Validation Dice', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Dice Score', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.set_title('Segmentation Dice Over Epochs', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved segmentation curves to {path}")

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
        