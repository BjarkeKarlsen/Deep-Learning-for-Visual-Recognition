import os
import math
from dataclasses import asdict
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode
from torch.amp import autocast, GradScaler

from deepfake.config import Config
from deepfake.utils.checkpoint_manager import CheckpointManager
from deepfake.data.dataset_manager import DatasetFilters, SIDDatasetManager, TRAIN, VALIDATION
from deepfake.utils.model_persister import TorchModelPersister, IModelPersister
from deepfake.data.dataset import SIDClassificationDataset
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.utils.optimizer_factory import OptimizerFactory
from deepfake.utils.logger import SidLogger
from deepfake.utils.training_metrics_tracker import TrainingMetrics, TrainingMetricsTracker
from deepfake.visualization.segmentation_plots import SegmentationPlots
from .dice_coefficient import dice_coefficient

class Trainer:
    """
    Class-based trainer for tamper segmentation with mixed precision,
    metrics tracking, checkpointing, and plotting.
    """

    def __init__(
        self,
        cfg: Config,
        logger: SidLogger,
        metrics_tracker: TrainingMetricsTracker,
        checkpoint_mgr: CheckpointManager,
        persister: IModelPersister = None
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.training.device)

        # Model, loss, optimizer, scaler
        self.model = TamperSegmentationModel(in_channels=3, out_channels=1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = OptimizerFactory(self.model.parameters(), cfg.training)
        self.scaler = GradScaler()

        # Persister, metrics, checkpoint
        self.persister = persister or TorchModelPersister()
        self.metrics_tracker = metrics_tracker
        self.checkpoint_mgr = checkpoint_mgr

        logger.log_training_config(asdict(cfg))

    def train(self, start_epoch: int = 0):
        """Train the tamper segmentation model on the configured SID subsets."""
        prev_best = None
        prev_best_val = float('-inf')
        
        # LOAD THE DATA
        train_loader, val_loader = self._load_data()

        # START TRAINING
        self.metrics_tracker.start_training()

        for epoch in range(start_epoch, self.cfg.training.epochs):
            # TRAIN & EVAL
            avg_train_loss, avg_train_dice = self._train_epoch(train_loader, epoch)
            avg_val_loss, avg_val_dice = self._evaluation(val_loader)
            
            # Determine previous best before recording current metrics
            prev_best = self.metrics_tracker.get_best_metric("val_dice")
            prev_best_val = prev_best.additional_metrics.get("val_dice") if prev_best and prev_best.additional_metrics.get("val_dice") is not None else float('-inf')


            # RECORD METRICS
            self.metrics_tracker.add_metrics(TrainingMetrics(
                epoch=epoch + 1,
                step=(epoch + 1) * len(train_loader),
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                additional_metrics={
                    "train_dice": avg_train_dice,
                    "val_dice": avg_val_dice,
                }
            ))
            
            # LOGGING EPOCH
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={avg_train_loss:.4f}, train_dice={avg_train_dice:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_dice={avg_val_dice:.4f}"
            )
            
            # SAVE BEST MODEL IF IMPROVED
            if avg_val_dice > prev_best_val:
                self.persister.save_model(self.model, self.cfg.paths.model_path)
                self.logger.info(f"Saved best model at epoch {epoch+1} (dice={avg_val_dice:.4f})")


            # CHECKPOINT
            self.checkpoint_mgr.create_checkpoint(
                epoch=epoch + 1,
                model=self.model,
                optimizer=self.optimizer,
                metrics_tracker=self.metrics_tracker,
                config=self.cfg,
            )

        # Finish
        self.metrics_tracker.end_training()
        self.metrics_tracker.save_to_json(self.cfg.paths.history_path)
            
        self.logger.log_training_complete(
            total_time=self.metrics_tracker.get_summary_stats().get('duration_minutes'),
            best_metric=self.metrics_tracker.get_best_metric("val_dice").additional_metrics.get("val_dice"),
            best_epoch=self.metrics_tracker.get_best_metric("val_dice").epoch,
        )
        
        # PLOT
        plotter = SegmentationPlots(output_directory=self.cfg.paths.run_root, training_history_path=self.cfg.paths.history_path)
        plotter.plot_training_history()
        plotter.plot_learning_rate_schedule()

            
    def _train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        train_loss, train_dice, steps = 0.0, 0.0, 0
        
        # ITERATE OVER BATCHES WITH A PROGRESS BAR
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.training.epochs}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # ZERO GRADIENTS BEFORE BACKWARD PASS
            self.optimizer.zero_grad()
            
            # ENABLE MIXED PRECISION FOR FASTER FORWARD AND REDUCED MEMORY
            with autocast(device_type=self.device.type):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
            
            # SCALE LOSS AND BACKWARD FOR STABLE MIXED-PRECISION TRAINING    
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # UPDATE TRAINING METRICS
            train_loss += loss.item()
            train_dice += dice_coefficient(logits, masks).item()
            
            steps += 1
            # UPDATE PROGRESS BAR WITH CURRENT LOSS
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'dice': f'{train_dice:.3f}'})
        
        # CALCULATE AVERAGE METRICS    
        avg_train_loss = train_loss / max(steps, 1)
        avg_train_dice = train_dice / max(steps, 1)
        
        return avg_train_loss, avg_train_dice

    @torch.no_grad() # Optimize memory usage and speed up computations
    def _evaluation(self, val_loader: DataLoader):
        
        self.model.eval()
        val_loss, val_dice = 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # FORWARD PASS THROUGH SEGMENTATION MODEL
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                
                # ACCUMULATE LOSS AND DICE SCORE
                val_loss += loss.item()
                val_dice += dice_coefficient(logits, masks).item()
        
        # CALCULATE AVERAGE METRICS ACROSS ALL VALIDATION BATCHES        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
        avg_val_dice = val_dice / len(val_loader) if len(val_loader) > 0 else float('nan')
        
        return avg_val_loss, avg_val_dice
    
    def _load_data(self):
        manager = SIDDatasetManager(
            dataset_name=self.cfg.data.dataset_name,
            use_streaming=self.cfg.data.use_streaming,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )
        train_ds = manager.get_split(
            split_type=TRAIN,
            max_samples=self.cfg.data.train_samples,
            filter_fn=DatasetFilters.tampered_with_masks
        )
        val_ds = manager.get_split(
            split_type=VALIDATION,
            max_samples=self.cfg.data.val_samples,
            filter_fn=DatasetFilters.tampered_with_masks
        )
        train_loader = DataLoader(
            SIDClassificationDataset(train_ds, image_size=self.cfg.data.image_size,
                                     normalize_mean=self.cfg.model.normalize_mean,
                                     normalize_std=self.cfg.model.normalize_std,
                                     return_mask=True),
            batch_size=self.cfg.loader.batch_size,
            shuffle=not self.cfg.data.use_streaming and self.cfg.loader.shuffle_train,
            num_workers=0 if self.cfg.data.use_streaming else self.cfg.loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            SIDClassificationDataset(val_ds, image_size=self.cfg.data.image_size,
                                     normalize_mean=self.cfg.model.normalize_mean,
                                     normalize_std=self.cfg.model.normalize_std,
                                     return_mask=True),
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=0 if self.cfg.data.use_streaming else self.cfg.loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader
            
        
        
