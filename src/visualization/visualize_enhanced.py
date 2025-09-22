
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_enhanced_training_curves(history, save_path='results/enhanced_training_curves.png'):
    """Enhanced training curves with comprehensive analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Training Analysis', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Overall Loss Comparison
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Overall Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Classification Performance
    axes[0, 1].plot(epochs, history['train_cls_loss'], 'g-', label='Train Cls Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['val_cls_loss'], 'orange', label='Val Cls Loss', linewidth=2)
    axes[0, 1].set_title('Classification Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Segmentation Performance  
    axes[0, 2].plot(epochs, history['train_seg_loss'], 'purple', label='Train Seg Loss', linewidth=2)
    axes[0, 2].plot(epochs, history['val_seg_loss'], 'brown', label='Val Seg Loss', linewidth=2)
    axes[0, 2].set_title('Segmentation Loss', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Accuracy Trends
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1, 0].set_title('Classification Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Segmentation Metrics
    if 'val_iou' in history and 'val_dice' in history:
        axes[1, 1].plot(epochs, history['val_iou'], 'cyan', label='IoU', linewidth=2)
        axes[1, 1].plot(epochs, history['val_dice'], 'magenta', label='Dice', linewidth=2)
        axes[1, 1].set_title('Segmentation Metrics', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Loss Component Analysis
    total_loss = [c + s for c, s in zip(history['train_cls_loss'], history['train_seg_loss'])]
    cls_ratio = [c/t for c, t in zip(history['train_cls_loss'], total_loss)]
    seg_ratio = [s/t for s, t in zip(history['train_seg_loss'], total_loss)]

    axes[1, 2].plot(epochs, cls_ratio, 'green', label='Cls Contribution', linewidth=2)
    axes[1, 2].plot(epochs, seg_ratio, 'purple', label='Seg Contribution', linewidth=2)
    axes[1, 2].set_title('Loss Component Ratio', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved enhanced training curves to {save_path}")

def create_performance_summary(history, save_path='results/performance_summary.png'):
    """Create a performance summary dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Summary Dashboard', fontsize=16, fontweight='bold')

    # Final metrics
    final_metrics = {
        'Train Acc': history['train_acc'][-1],
        'Val Acc': history['val_acc'][-1],
        'Train Loss': history['train_loss'][-1],
        'Val Loss': history['val_loss'][-1],
        'IoU': history['val_iou'][-1] if 'val_iou' in history else 0,
        'Dice': history['val_dice'][-1] if 'val_dice' in history else 0
    }

    # 1. Final Metrics Bar Chart
    metrics_names = list(final_metrics.keys())
    metrics_values = list(final_metrics.values())
    colors = ['green' if 'Acc' in name or name in ['IoU', 'Dice'] else 'red' for name in metrics_names]

    axes[0, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Final Performance Metrics', fontweight='bold')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for i, v in enumerate(metrics_values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Learning Progress
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 1].plot(epochs, history['val_loss'], 'r-', linewidth=3, label='Val Loss')
    axes[0, 1].fill_between(epochs, history['val_loss'], alpha=0.3, color='red')
    axes[0, 1].set_title('Validation Loss Progress', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Overfitting Analysis
    train_val_diff = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
    axes[1, 0].plot(epochs, train_val_diff, 'orange', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(epochs, train_val_diff, 0, 
                           where=[diff > 0 for diff in train_val_diff], 
                           color='red', alpha=0.3, label='Overfitting')
    axes[1, 0].fill_between(epochs, train_val_diff, 0,
                           where=[diff <= 0 for diff in train_val_diff],
                           color='green', alpha=0.3, label='Good Fit')
    axes[1, 0].set_title('Overfitting Analysis (Val - Train Loss)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Segmentation Progress
    if 'val_iou' in history:
        iou_improvement = [iou - history['val_iou'][0] for iou in history['val_iou']]
        axes[1, 1].plot(epochs, iou_improvement, 'blue', linewidth=2, marker='o', markersize=4)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(epochs, iou_improvement, 0, alpha=0.3, 
                               color='blue' if iou_improvement[-1] > 0 else 'red')
        axes[1, 1].set_title('IoU Improvement Over Training', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU Change')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved performance summary to {save_path}")

def analyze_training_problems(history):
    """Analyze training and return diagnostics"""
    problems = []
    recommendations = []

    # Check overfitting
    final_val_loss = history['val_loss'][-1]
    final_train_loss = history['train_loss'][-1]
    if final_val_loss > final_train_loss * 1.5:
        problems.append("🚨 Severe overfitting detected")
        recommendations.append("• Reduce model complexity or add regularization")
        recommendations.append("• Add more training data or data augmentation")

    # Check segmentation learning
    if 'val_iou' in history:
        initial_iou = history['val_iou'][0] if len(history['val_iou']) > 0 else 0
        final_iou = history['val_iou'][-1]
        if final_iou < 0.1:
            problems.append("🚨 Segmentation completely failed")
            recommendations.append("• Check mask preprocessing and alignment")
            recommendations.append("• Increase segmentation loss weight significantly")
            recommendations.append("• Verify data loading pipeline")
        elif final_iou < initial_iou + 0.01:
            problems.append("⚠️ Segmentation not improving")
            recommendations.append("• Try different segmentation loss function")
            recommendations.append("• Adjust loss weights")

    # Check classification performance
    final_val_acc = history['val_acc'][-1]
    if final_val_acc < 60:
        problems.append("⚠️ Poor classification performance")
        recommendations.append("• Check data quality and class balance")
        recommendations.append("• Consider using class weights")

    # Check convergence
    if len(history['train_loss']) > 10:
        recent_loss = history['train_loss'][-5:]
        if max(recent_loss) - min(recent_loss) > 0.5:
            problems.append("⚠️ Training unstable/not converged")
            recommendations.append("• Reduce learning rate")
            recommendations.append("• Add gradient clipping")

    return problems, recommendations

# def create_diagnostic_report(history, save_path='results/diagnostic_report.txt'):
#     """Create a text-based diagnostic report"""
#     problems, recommendations = analyze_training_problems(history)

#     report = f"""
# DEEPFAKE DETECTION MODEL DIAGNOSTIC REPORT
# ==========================================

# TRAINING SUMMARY:
# - Total Epochs: {len(history['train_loss'])}
# - Final Train Loss: {history['train_loss'][-1]:.4f}
# - Final Val Loss: {history['val_loss'][-1]:.4f}
# - Final Train Accuracy: {history['train_acc'][-1]:.1f}%
# - Final Val Accuracy: {history['val_acc'][-1]:.1f}%
# """

#     if 'val_iou' in history:
#         report += f"- Final IoU: {history['val_iou'][-1]:.6f}

#         report += f"- Final Dice: {history['val_dice'][-1]:.6f}


#     report += f"PROBLEMS IDENTIFIED:
# "
#     if problems:
#         for problem in problems:
#             report += f"{problem}
# "
#     else:
#         report += "✅ No major problems detected!
# "

#     report += f"
# RECOMMENDATIONS:
# "
#     if recommendations:
#         for rec in recommendations:
#             report += f"{rec}
# "
#     else:
#         report += "✅ Model performance looks good!
# "

#     # Calculate some additional metrics
#     overfitting_ratio = history['val_loss'][-1] / history['train_loss'][-1]
#     report += f"
# ADDITIONAL METRICS:
# "
#     report += f"- Overfitting Ratio (Val/Train Loss): {overfitting_ratio:.2f}
# "
#     report += f"- Loss Reduction: {history['train_loss'][0] - history['train_loss'][-1]:.4f}
# "

#     if 'val_iou' in history and len(history['val_iou']) > 0:
#         iou_improvement = history['val_iou'][-1] - history['val_iou'][0]
#         report += f"- IoU Improvement: {iou_improvement:.6f}
# "

#     with open(save_path, 'w') as f:
#         f.write(report)

#     print(f"✅ Saved diagnostic report to {save_path}")
#     return problems, recommendations
