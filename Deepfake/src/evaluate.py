import os, sys, torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import SIDDataset
from .model import SimpleCNN
from .visualize import plot_classification_metrics, plot_confusion_matrix
from .utils.model_manager import check_model_exists, save_training_history
from .dataset_manager import SIDDatasetManager
from .utils.logger import SidLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def evaluate(logger: SidLogger, cfg: Config):
    device = torch.device(cfg.training.device)

    # Log evaluation configuration
    logger.log_evaluation_config(cfg.to_dict())

    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)

    check_model_exists(cfg.paths.model_path)

    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_disk_cache=cfg.data.use_disk_cache,
        use_streaming=cfg.data.use_streaming
    )
    _, _, test_ds = manager.get_splits(test_max=cfg.data.test_samples)

    test_loader = DataLoader(
        SIDDataset(
            test_ds,
            image_size=cfg.model.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std,
            device=device
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_test,
        num_workers=cfg.loader.num_workers
    )

    # Load model
    model = SimpleCNN(num_classes=cfg.model.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.paths.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    logger.info("Starting evaluation...")
    with logger.time_block("model inference"):
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

    # Compute metrics
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")

    # Classification report
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=list(cfg.model.class_names),
        output_dict=True
    )
    logger.log_classification_report(report_dict)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.log_confusion_matrix(cm, labels=list(cfg.model.class_names))

    # Save raw results JSON
    results = {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    logger.save_json(results, "evaluation_results.json")

    # Save training history file
    save_training_history(cfg.paths.results_dir, results, history_file="evaluation.json")

    logger.info("Generating visualizations...")
    plot_confusion_matrix(cm, cfg.model.class_names)
    plot_classification_metrics(report_dict)

    logger.info("Evaluation complete.")
