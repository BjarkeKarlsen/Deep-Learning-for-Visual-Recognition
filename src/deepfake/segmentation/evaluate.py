import math
from dataclasses import asdict
import os
from typing import Dict

from deepfake.data.dataset import SIDClassificationDataset
from deepfake.segmentation.dice_and_iou import dice_and_iou
from deepfake.visualization.segmentation_plots import SegmentationPlots
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode

from deepfake.data.dataset_manager import TEST, DatasetFilters, SIDDatasetManager
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.utils.evaluation_metrics_tracker import SegmentationEvaluationMetrics, EvaluationMetricsTracker
from deepfake.utils.logger import SidLogger
from deepfake.utils.model_persister import TorchModelPersister, IModelPersister
from deepfake.config import Config


class Evaluator:
    """
    Runs segmentation inference, records Dice/IoU metrics, and generates gallery and plots.
    """

    def __init__(
        self,
        cfg: Config,
        logger: SidLogger,
        persister: IModelPersister = None
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.training.device)

        # Dataset & DataLoader
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
            filter_fn=DatasetFilters.tampered_with_masks,
        )
        self.test_loader = DataLoader(
            SIDClassificationDataset(
                test_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                return_mask=True,
            ),
            batch_size=cfg.loader.batch_size,
            shuffle=False,
            num_workers=cfg.loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Metrics tracker
        self.metrics_tracker = EvaluationMetricsTracker(
            SegmentationEvaluationMetrics,
            logger=logger,
        )

        # Model
        self.model = TamperSegmentationModel(in_channels=3, out_channels=1).to(self.device)
        self.persister = persister or TorchModelPersister()
        self.persister.load_model(self.model, cfg.paths.model_path)
        self.logger.info(f"Loaded segmentation model from {cfg.paths.model_path}")

    def run(self):
        """Full evaluation pipeline."""
        self.logger.log_evaluation_config(asdict(self.cfg))
        dice_scores, iou_scores, gallery = self.infer()
        self.compute_and_log(dice_scores, iou_scores)
        self.save_and_plot(gallery)
        self.logger.info("Segmentation evaluation complete")

    @torch.no_grad()
    def infer(self):
        """Run inference and collect dice & iou per batch, plus gallery examples."""
        self.model.eval()
        dice_scores, iou_scores = [], []
        gallery = []
        max_gallery = 6

        self.logger.info("Starting segmentation evaluation")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Eval"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                logits = self.model(images)
                metrics = dice_and_iou(logits, masks)
                dice_scores.append(metrics["dice"].item())
                iou_scores.append(metrics["iou"].item())

                if len(gallery) < max_gallery:
                    img = images[0].cpu()
                    mean = torch.tensor(self.cfg.model.normalize_mean).view(3,1,1)
                    std = torch.tensor(self.cfg.model.normalize_std).view(3,1,1)
                    denorm = (img * std + mean).clamp(0,1).permute(1,2,0).numpy()
                    true_mask = masks[0].squeeze(0).cpu().numpy()
                    pred_mask = (torch.sigmoid(logits[0])>0.5).squeeze(0).cpu().numpy().astype(float)
                    gallery.append((denorm, true_mask, pred_mask))

        return dice_scores, iou_scores, gallery

    def compute_and_log(self, dice_scores, iou_scores):
        """Compute means, log, and record metrics."""
        mean_dice = float(np.mean(dice_scores)) if dice_scores else float("nan")
        mean_iou = float(np.mean(iou_scores)) if iou_scores else float("nan")
        self.logger.info(f"Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}")

        self.metrics_tracker.add_metrics(
            SegmentationEvaluationMetrics(
                task_type="segmentation",
                primary_metric="dice",
                primary_score=mean_dice,
                dice_coefficient=mean_dice,
                mean_iou=mean_iou,
            )
        )

    def save_and_plot(self, gallery):
        """Persist metrics, save gallery, and generate plots."""
        metrics_path = self.cfg.paths.metrics_path
        self.metrics_tracker.save_to_json(metrics_path)
        self.logger.info(f"Saved evaluation metrics to {metrics_path}")

        plotter = SegmentationPlots(
            output_directory=self.cfg.paths.run_root,
            eval_history_path=metrics_path,
        )
        plotter.plot_segmentation_gallery(gallery, ncols=3, save_path=str(self.cfg.paths.run_root / "segmentation_gallery.png"))
        plotter.plot_training_history()  # plots history of dice/IoU
        self.logger.info("Generated segmentation evaluation plots")

