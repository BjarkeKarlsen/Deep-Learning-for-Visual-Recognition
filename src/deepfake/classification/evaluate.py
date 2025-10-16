import torch
import numpy as np
import os
from dataclasses import asdict
from datasets import DownloadMode

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from deepfake.utils.model_persister import TorchModelPersister
from deepfake.visualization.classification_plots import ClassificationPlots
from deepfake.utils.evaluation_metrics_tracker import ClassificationEvaluationMetrics, EvaluationMetricsTracker
from deepfake.config import Config
from deepfake.data.dataset_manager import TEST, SIDDatasetManager
from deepfake.utils.model_manager import check_model_exists
from deepfake.utils.logger import SidLogger

from ..data.dataset import SIDClassificationDataset
from .model import BaselineClassifier


def evaluate(logger: SidLogger, cfg: Config):
    """Run classification inference and report accuracy/diagnostics."""
    device = torch.device(cfg.training.device)

    logger.log_evaluation_config(asdict(cfg))

    if not check_model_exists(cfg.paths.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {cfg.paths.model_path}")

    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_streaming=cfg.data.use_streaming,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    
    test_ds = manager.get_split(
            split_type=TEST,
            max_samples=cfg.data.test_samples,
            use_test_or_val_as_test_set=True,
            val_offset=cfg.data.val_samples,
    )

    test_loader = DataLoader(
        SIDClassificationDataset(
            test_ds,
            image_size=cfg.data.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std,
            return_label=True,
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_test,
        num_workers=cfg.loader.num_workers,
    )

    metrics_tracker = EvaluationMetricsTracker(ClassificationEvaluationMetrics, logger=logger)

    model = BaselineClassifier(num_classes=cfg.model.num_classes).to(device)
    model_persister = TorchModelPersister()
    model_persister.load_model(
        model,
        cfg.paths.model_path,
        device=cfg.training.device,
    )
    model.eval()
    
    all_preds = []
    all_labels = []

    logger.info("Starting evaluation...")
    with logger.time_block("model inference"):
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Cls Eval")
            for batch in pbar:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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

    metrics_tracker.save_to_json(cfg.paths.metrics_path)

    logger.info("Generating visualizations...")
    classification_plotter = ClassificationPlots(
        eval_history_path=cfg.paths.metrics_path,
        output_directory=cfg.paths.run_root
    )
    
    classification_plotter.plot_confusion_matrix()
    classification_plotter.plot_classification_report()

    logger.info("Evaluation complete.")
