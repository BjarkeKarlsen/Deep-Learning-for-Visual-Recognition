import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names.values(),
                yticklabels=class_names.values())
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_classification_metrics(report, save_path='results/metrics_barplot.png'):
    classes = ['Real', 'Synthetic', 'Tampered']
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
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot to {save_path}")