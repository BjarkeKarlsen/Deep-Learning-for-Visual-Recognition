import torch
import numpy as np
import os
from dataclasses import asdict

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from deepfake.utils.model_persister import TorchModelPersister
from deepfake.visualization.classification_plots import ClassificationPlots
from deepfake.utils.evaluation_metrics_tracker import ClassificationEvaluationMetrics, EvaluationMetricsTracker
from deepfake.config import Config
from deepfake.data.dataset_manager import SIDDatasetManager
from deepfake.utils.model_manager import check_model_exists
from deepfake.utils.logger import SidLogger

from .dataset import SIDClassificationDataset
from .model import BaselineClassifier


def evaluate(logger: SidLogger, cfg: Config):
    """Run classification inference and report accuracy/diagnostics."""
    device = torch.device(cfg.training.device)

    logger.log_evaluation_config(asdict(cfg))

    check_model_exists(cfg.paths.model_path)

    with SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_disk_cache=cfg.data.use_disk_cache,
        use_streaming=cfg.data.use_streaming,
    ) as manager:
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
            normalize_std=cfg.model.normalize_std,
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_test,
        num_workers=cfg.loader.num_workers,
    )

    metrics_tracker = EvaluationMetricsTracker(ClassificationEvaluationMetrics, logger=logger)

    model = BaselineClassifier(num_classes=cfg.model.num_classes).to(device)
    model_persister = TorchModelPersister()
    model_persister.load_model(model, cfg.paths.model_path)
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

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=list(cfg.model.class_names),
        output_dict=True,
        zero_division=0 
    )
    logger.log_classification_report(report_dict)

    cm = confusion_matrix(all_labels, all_preds)
    logger.log_confusion_matrix(cm, labels=list(cfg.model.class_names))

    # Instead of passing a dict, create a proper metrics object:
    metrics_tracker.add_metrics(ClassificationEvaluationMetrics(
        task_type="classification",
        primary_metric="accuracy",
        primary_score=accuracy / 100.0,  # Convert percentage to decimal
        accuracy=accuracy / 100.0,
        classification_report=report_dict,
        confusion_matrix=cm.tolist(),
        class_names=list(cfg.model.class_names)
    ))

    metrics_tracker.save_to_json(cfg.paths.eval_metrics_path)

    logger.info("Generating visualizations...")
    classification_plotter = ClassificationPlots(
        eval_history_path=cfg.paths.eval_metrics_path,
        output_directory=cfg.paths.output_dir
    )
    
    classification_plotter.plot_confusion_matrix(save_path=cfg.paths.output_dir)
    classification_plotter.plot_classification_report(save_path=cfg.paths.output_dir)

    logger.info("Evaluation complete.")
