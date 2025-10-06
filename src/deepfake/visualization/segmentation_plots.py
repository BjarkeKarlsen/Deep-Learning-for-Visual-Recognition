from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

from deepfake.visualization.common_plots import CommonPlots

def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.5,
                 color: Tuple[float,float,float] = (1,0,0)) -> np.ndarray:
    """
    Return the image with mask overlaid in the given RGB color.
    image: H×W×3 float [0–1] or uint8 [0–255]
    mask:  H×W binary or float probability
    """
    # Ensure uint8 [0..255]
    img = (image*255).astype(np.uint8) if image.dtype != np.uint8 else image.copy()
    # Build color overlay
    overlay = np.zeros_like(img)
    overlay[mask>0] = (np.array(color)*255).astype(np.uint8)
    return cv2.addWeighted(img, 1.0, overlay, alpha, 0)



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

    def plot_segmentation(self, image: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot the input image, the ground-truth mask, and a predicted-mask overlay.

        Args:
            image (H×W×C or H×W numpy): Input image.
            true_mask (H×W numpy): Ground-truth segmentation (0/1).
            pred_mask (H×W numpy): Predicted segmentation (0/1 or probabilities).
            save_path: Optional path to save the figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Input image
        axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[0].set_title('Image'); axes[0].axis('off')

        # Ground-truth mask
        axes[1].imshow(true_mask, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('True Mask'); axes[1].axis('off')

        # Predicted overlay
        #axes[2].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[2].imshow(pred_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Predicted Overlay'); axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved segmentation plot to {save_path}")
        else:
            plt.show()
        plt.close()
        

    def plot_segmentation_gallery(self,
                          examples: List[Tuple[np.ndarray,np.ndarray,np.ndarray]],
                          ncols: int = 3,
                          save_path: Optional[str] = None) -> None:
        """
        Plot a grid of segmentation examples.
        examples: list of (image, true_mask, pred_mask)
        ncols: number of columns per row (each example uses 3 subplots)
        """
        n = len(examples)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows*2, ncols, figsize=(4*ncols, 4*nrows*2))
        axes = axes.reshape(nrows*2, ncols)

        for idx, (img, true_m, pred_m) in enumerate(examples):
            row = (idx // ncols)*2
            col = idx % ncols

            # True overlay
            ax = axes[row, col]
            ax.imshow(overlay_mask(img, true_m, color=(0,1,0)))
            ax.set_title("True Mask Overlay")
            ax.axis("off")

            # Pred overlay
            ax = axes[row+1, col]
            ax.imshow(overlay_mask(img, pred_m, color=(1,0,0)))
            ax.set_title("Pred Mask Overlay")
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved segmentation gallery to {save_path}")
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
        