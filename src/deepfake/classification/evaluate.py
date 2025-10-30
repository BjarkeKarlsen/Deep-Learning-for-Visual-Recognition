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
from deepfake.utils.augmentation_factory import build_classification_transform
from deepfake.utils.evaluation_metrics_tracker import ClassificationEvaluationMetrics, EvaluationMetricsTracker
from deepfake.config import Config
from deepfake.data.dataset_manager import TEST, SIDDatasetManager
from deepfake.utils.logger import SidLogger

from ..data.dataset import SIDClassificationDataset
from .model import BaselineClassifier

class Evaluator:
    """
    Runs classification inference, records metrics, and generates plots.
    """
    # HANDLES TEST-TIME INFERENCE, METRIC REPORTING, AND PLOT GENERATION FOR CLASSIFICATION RUNS.

    def __init__(self, cfg: Config, logger: SidLogger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.training.device)

        # DATASET & DATALOADER
        # BUILD A DETERMINISTIC TEST LOADER THAT USES THE SAME PREPROCESSING AS TRAINING.
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
        test_transform = build_classification_transform(
            cfg.data.image_size,
            cfg.model.normalize_mean,
            cfg.model.normalize_std,
            cfg.data.augment,
            is_train=False,
        )
        self.test_loader = DataLoader(
            SIDClassificationDataset(
                test_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                transform=test_transform,
                return_label=True,
            ),
            batch_size=cfg.loader.batch_size,
            shuffle=cfg.loader.shuffle_test,
            num_workers=cfg.loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # METRICS TRACKER
        # COLLECT ALL EVALUATION RESULTS SO THEY CAN BE SAVED AND PLOTTED LATER.
        self.metrics_tracker = EvaluationMetricsTracker(
            ClassificationEvaluationMetrics,
            logger,
        )

        # MODEL
        # RESTORE THE BEST-SAVED WEIGHTS TO THE TARGET DEVICE BEFORE INFERENCE.
        self.model = BaselineClassifier(num_classes=cfg.model.num_classes).to(self.device)
        self.persister = TorchModelPersister()
        self.persister.load_model(self.model, self.cfg.paths.model_path, device=self.device)
        self.logger.info(f"Loaded model from {self.cfg.paths.model_path}")


    def run(self):
        """Execute the full evaluation pipeline."""
        self.logger.log_evaluation_config(asdict(self.cfg))
        # MAIN ENTRYPOINT: RUN INFERENCE, SUMMARISE METRICS, THEN GENERATE PLOTS.
        labels, preds = self.infer()
        self.compute_and_log(labels, preds)
        self.save_and_plot()
        self.logger.info("Evaluation complete")


    @torch.no_grad()
    def infer(self):
        """Run inference over the test set and collect predictions."""
        self.model.eval()
        all_preds, all_labels = [], []
        self.logger.info("Starting evaluation inference")
        for batch in tqdm(self.test_loader, desc="Eval"):
            # ACCUMULATE MODEL PREDICTIONS AND TRUE LABELS BATCH BY BATCH.
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        return all_labels, all_preds

    def compute_and_log(self, labels, preds):
        """Compute accuracy, classification report, confusion matrix, and log them."""
        accuracy = float(np.mean(np.array(preds) == np.array(labels)))
        self.logger.info(f"Overall Accuracy: {accuracy:.4f}")

        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(
            labels,
            preds,
            target_names=self.cfg.model.class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds)

        self.logger.log_classification_report(report)
        self.logger.log_confusion_matrix(cm, labels=self.cfg.model.class_names)

        # RECORD EVALUATION METRICS
        # STORE STRUCTURED RESULTS SO LATER COMMANDS (PLOT, ANALYTICS) CAN REUSE THEM.
        self.metrics_tracker.add_metrics(
            ClassificationEvaluationMetrics(
                task_type="classification",
                primary_metric="accuracy",
                primary_score=accuracy,
                accuracy=accuracy,
                classification_report=report,
                confusion_matrix=cm.tolist(),
                class_names=self.cfg.model.class_names,
            )
        )

    def save_and_plot(self):
        """Save metrics JSON and generate evaluation plots."""
        self.metrics_tracker.save_to_json(self.cfg.paths.metrics_path)
        self.logger.info(f"Saved evaluation metrics to {self.cfg.paths.metrics_path}")

        plotter = ClassificationPlots(
            eval_history_path=self.cfg.paths.metrics_path,
            output_directory=self.cfg.paths.run_root,
        )
        plotter.plot_confusion_matrix()
        plotter.plot_classification_report()
        self.logger.info("Generated evaluation plots")
