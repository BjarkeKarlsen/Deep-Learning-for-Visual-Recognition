from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Input image
        axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[0].set_title('Image')
        axes[0].axis('off')

        # Ground-truth mask
        axes[1].imshow(true_mask, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('True Mask')
        axes[1].axis('off')

        # Predicted overlay
        axes[2].imshow(pred_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Predicted Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        
        # Use the base class save_plot method
        self.save_plot(fig, filename=filename or "segmentation_plot.png", save_path=save_path)

    def plot_segmentation_gallery(self,
                                 examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 ncols: int = 3,
                                 filename: Optional[str] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot a grid of segmentation examples.
        
        Args:
            examples: list of (image, true_mask, pred_mask) tuples
            ncols: number of columns per row (each example uses 3 subplots)
            filename: Optional filename for the saved plot.
            save_path: Optional directory path to save the figure.
        """
        n = len(examples)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 4 * nrows * 2))
        axes = axes.reshape(nrows * 2, ncols)

        for idx, (img, true_m, pred_m) in enumerate(examples):
            row = (idx // ncols) * 2
            col = idx % ncols

            # True overlay
            ax = axes[row, col]
            ax.imshow(self.overlay_mask(img, true_m, color=(0, 1, 0)))
            ax.set_title("True Mask Overlay")
            ax.axis("off")

            # Pred overlay
            ax = axes[row + 1, col]
            ax.imshow(self.overlay_mask(img, pred_m, color=(1, 0, 0)))
            ax.set_title("Pred Mask Overlay")
            ax.axis("off")

        # Hide any unused axes so the grid does not show empty frames.
        total_slots = axes.size
        used_slots = len(examples) * 2
        if used_slots < total_slots:
            for ax in axes.flatten()[used_slots:]:
                ax.axis("off")

        plt.tight_layout()
        
        # Use the base class save_plot method
        self.save_plot(fig, filename=filename or "segmentation_gallery.png", save_path=save_path)

    def plot_training_history(self, filename: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """Plot segmentation loss and Dice curves extracted from the history dictionary."""
        if self.history is None:
            print("No training history loaded. Cannot plot training curves.")
            return
            
        curves = self.history

        # Number of epochs based on train_loss length
        n_epochs = len(curves.get('train_loss', []))
        if n_epochs == 0:
            print("No training loss data found in history.")
            return
            
        epochs = range(1, n_epochs + 1)

        # Handle dice metrics if available
        raw_train_dice = curves.get('train_dice', [])
        raw_val_dice = curves.get('val_dice', [])
        
        # Pad dice arrays to match epochs length
        train_dice = list(raw_train_dice) + [np.nan] * (n_epochs - len(raw_train_dice))
        val_dice = list(raw_val_dice) + [np.nan] * (n_epochs - len(raw_val_dice))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Loss plot
        ax1.plot(epochs, curves.get('train_loss', []), 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in curves and curves['val_loss']:
            val_loss = curves.get('val_loss', [np.nan] * n_epochs)
            # Pad validation loss if needed
            if len(val_loss) < n_epochs:
                val_loss = list(val_loss) + [np.nan] * (n_epochs - len(val_loss))
            ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Segmentation Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Dice plot
        if raw_train_dice or raw_val_dice:
            ax2.plot(epochs, train_dice, 'g-', label='Training Dice', linewidth=2)
            ax2.plot(epochs, val_dice, 'm-', label='Validation Dice', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Dice Score', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.set_title('Segmentation Dice Over Epochs', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No Dice score data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Segmentation Dice Over Epochs', fontsize=14)

        plt.tight_layout()
        
        # Use the base class save_plot method
        self.save_plot(fig, filename=filename or "training_history.png", save_path=save_path)

    def plot_learning_rate_schedule(self, filename: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """Plot LR schedule from self.history."""
        if self.history is None:
            print("No training history loaded. Cannot plot learning rate schedule.")
            return
            
        curves = self.history
        if not curves.get("learning_rate"):
            print("No learning_rate data to plot")
            return
            
        lr = curves["learning_rate"]
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
        
        # Use the base class save_plot method
        self.save_plot(fig, filename=filename or "learning_rate_schedule.png", save_path=save_path)

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
        # Ensure uint8 [0..255]
        img = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image.copy()
        
        # Build color overlay
        overlay = np.zeros_like(img)
        overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)
        
        return cv2.addWeighted(img, 1.0, overlay, alpha, 0)
