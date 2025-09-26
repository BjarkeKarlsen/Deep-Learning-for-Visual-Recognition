import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def _resolve_path(save_path, output_dir, default_filename):
    """Build the final path using the provided save_path or output_dir."""
    if save_path:
        return Path(save_path)
    if output_dir:
        return Path(output_dir) / default_filename
    return Path("results") / default_filename


def plot_confusion_matrix(cm, class_names, *, save_path=None, output_dir=None):
    path = _resolve_path(save_path, output_dir, "confusion_matrix.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_classification_metrics(report, *, class_names, save_path=None, output_dir=None):
    path = _resolve_path(save_path, output_dir, "metrics_barplot.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(class_names)
    metrics = ['precision', 'recall', 'f1-score']

    values = {metric: [] for metric in metrics}
    for cls in classes:
        for metric in metrics:
            values[metric].append(report[cls][metric])

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, values['precision'], width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x, values['recall'], width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + width, values['f1-score'], width, label='F1-Score', color='#F18F01')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Performance by Class', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot to {path}")


def plot_training_curves(history, *, save_path=None, output_dir=None):
    path = _resolve_path(save_path, output_dir, "training_curves.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss Over Epochs', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy Over Epochs', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {path}")


def plot_segmentation_curves(history, *, save_path=None, output_dir=None):
    path = _resolve_path(save_path, output_dir, "segmentation_curves.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = range(1, len(history.get('train_loss', [])) + 1)

    ax1.plot(epochs, history.get('train_loss', []), 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Segmentation Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history.get('train_dice', []), 'g-', label='Training Dice', linewidth=2)
    if 'val_dice' in history and history['val_dice']:
        ax2.plot(epochs, history['val_dice'], 'm-', label='Validation Dice', linewidth=2)
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
