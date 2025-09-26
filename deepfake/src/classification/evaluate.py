import torch
import numpy as np
import pandas as pd
from dataclasses import asdict

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import Config
from src.common.dataset_manager import SIDDatasetManager
from src.common.visualize import (
    plot_classification_metrics,
    plot_confusion_matrix,
)
from src.utils.model_manager import check_model_exists, save_training_history
from src.utils.logger import SidLogger

from .dataset import SIDClassificationDataset
from .model import BaselineClassifier


def evaluate(logger: SidLogger, cfg: Config):
    """Run classification inference and report accuracy/diagnostics."""
    device = torch.device(cfg.training.device)

    # Log evaluation configuration
    logger.log_evaluation_config(asdict(cfg))

    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)

    check_model_exists(cfg.paths.model_path)

    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_disk_cache=cfg.data.use_disk_cache,
        use_streaming=cfg.data.use_streaming
    )
    _, _, test_ds = manager.get_splits(
        train_max=0,
        val_max=0,
        test_max=cfg.data.test_samples,
        test_offset=cfg.data.val_samples,
    )

    test_loader = DataLoader(
        SIDClassificationDataset(
            test_ds,
            image_size=cfg.data.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_test,
        num_workers=cfg.loader.num_workers
    )

    # Load model
    model = BaselineClassifier(num_classes=cfg.model.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.paths.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    logger.info("Starting evaluation...")
    with logger.time_block("model inference"):
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                images = batch["image"].to(device)
                labels = batch["label"]
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
    plot_confusion_matrix(
        cm,
        cfg.model.class_names,
        output_dir=cfg.paths.results_dir,
    )
    plot_classification_metrics(
        report_dict,
        class_names=cfg.model.class_names,
        output_dir=cfg.paths.results_dir,
    )

    logger.info("Evaluation complete.")
