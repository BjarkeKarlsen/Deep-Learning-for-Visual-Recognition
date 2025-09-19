import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import SIDDataset
from .model import SimpleCNN
from .utils import get_device, print_gpu_info
from .visualize import plot_confusion_matrix, plot_classification_metrics
from config import *

def evaluate():
    device = get_device()

    print("="*60)
    print("EVALUATION MODE")
    print("="*60)
    print_gpu_info(device)
    print("="*60)

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_dataset = SIDDataset(split='test', max_samples=TEST_SAMPLES)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.1f}%")

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds,
                                   target_names=list(CLASS_NAMES.values()),
                                   output_dict=True)
    print(classification_report(all_labels, all_preds,
                                target_names=list(CLASS_NAMES.values())))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print("      Real  Syn  Tamp")
    for i, row in enumerate(cm):
        print(f"{list(CLASS_NAMES.values())[i]:5s} {row[0]:4d} {row[1]:4d} {row[2]:4d}")

    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'evaluation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/evaluation.json")

    print("\nGenerating visualizations...")
    plot_confusion_matrix(cm, CLASS_NAMES)
    plot_classification_metrics(report)

    print("="*60)

if __name__ == "__main__":
    evaluate()