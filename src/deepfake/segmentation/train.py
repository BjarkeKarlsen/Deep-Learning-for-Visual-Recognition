from dataclasses import asdict
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
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
from deepfake.utils.scheduler_factory import SchedulerFactory
from deepfake.visualization.segmentation_plots import SegmentationPlots
from .dice_coefficient import dice_coefficient, soft_dice_loss
from deepfake.utils.augmentation_factory import build_segmentation_transforms

class Trainer:
    """
    Class-based trainer for tamper segmentation with mixed precision,
    metrics tracking, checkpointing, and plotting.
    """
    # COORDINATES SEGMENTATION TRAINING, LOGGING, AND CHECKPOINTING.

    def __init__(
        self,
        cfg: Config,
        logger: SidLogger,
        metrics_tracker: TrainingMetricsTracker,
        checkpoint_mgr: CheckpointManager,
        persister: IModelPersister = None
    ):
        self.cfg = cfg
        # KEEP REFERENCES TO LOGGING, METRICS, AND CHECKPOINT HELPERS.
        self.logger = logger
        self.device = torch.device(cfg.training.device)

        # MODEL, LOSS, OPTIMISER, AND AMP HELPERS FOR SEGMENTATION.
        self.model = TamperSegmentationModel(model_cfg=cfg.model, in_channels=3, out_channels=1).to(self.device)
        self._log_model_summary()
        self.bce_loss = nn.BCEWithLogitsLoss()
        loss_cfg = getattr(cfg.training, "loss", None)
        self.bce_weight = getattr(loss_cfg, "bce_weight", 0.5)
        self.dice_weight = getattr(loss_cfg, "dice_weight", 0.5)
        total_weight = self.bce_weight + self.dice_weight
        if total_weight > 0:
            self.bce_weight /= total_weight
            self.dice_weight /= total_weight
        self.optimizer = OptimizerFactory(self.model.parameters(), cfg.training)
        self.amp_enabled = (self.device.type == "cuda" and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.scheduler = None
        self.scheduler_step_mode = "epoch"

        # Persister, metrics, checkpoint
        self.persister = persister or TorchModelPersister()
        self.metrics_tracker = metrics_tracker
        self.checkpoint_mgr = checkpoint_mgr

        metric_directions = {
            "train_dice": "max",
            "val_dice": "max",
            "train_bce_loss": "min",
            "train_dice_loss": "min",
            "val_bce_loss": "min",
            "val_dice_loss": "min",
        }
        for idx, _ in enumerate(self.optimizer.param_groups):
            metric_directions[f"lr_group_{idx}"] = "last"
        self.metrics_tracker.configure_tracking(
            track_train_acc=False,
            track_val_acc=False,
            metric_directions=metric_directions,
        )

        self._log_backbone_trainability()
        logger.log_training_config(asdict(cfg))

    def train(self, start_epoch: int = 0):
        """Train the tamper segmentation model on the configured SID subsets."""
        prev_best = None
        prev_best_val = float('-inf')
        
        # PREPARE LOADER PAIRS FILTERED TO TAMPERED SAMPLES WITH MASKS.
        train_loader, val_loader = self._load_data()
        self._log_forward_shape_snapshot(train_loader)

        try:
            steps_per_epoch = len(train_loader)
        except (TypeError, AttributeError):
            steps_per_epoch = 0
        # Instantiate LR scheduler (cosine/onecycle) if requested
        self.scheduler, self.scheduler_step_mode = SchedulerFactory.create(
            self.optimizer,
            self.cfg.training,
            steps_per_epoch=steps_per_epoch,
        )
        if self.scheduler and self.scheduler_step_mode == "epoch" and start_epoch > 0:
            self.scheduler.last_epoch = start_epoch - 1

        # START TRACKING TO CAPTURE TIMESTAMPS AND INITIAL HISTORY.
        self.metrics_tracker.start_training()
        base_step = self.metrics_tracker.metrics[-1].step if self.metrics_tracker.metrics else 0

        # MAIN LOOP HANDLES TRAINING STEPS, VALIDATION, AND CHECKPOINTS.
        for epoch in range(start_epoch, self.cfg.training.epochs):
            (
                avg_train_loss,
                avg_train_dice,
                steps_this_epoch,
                avg_train_bce_loss,
                avg_train_dice_loss,
            ) = self._train_epoch(train_loader, epoch)
            avg_val_loss, avg_val_dice, avg_val_bce_loss, avg_val_dice_loss = self._evaluation(val_loader)
            
            # Determine previous best before recording current metrics
            prev_best = self.metrics_tracker.get_best_metric("val_dice")
            prev_best_val = prev_best.additional_metrics.get("val_dice") if prev_best and prev_best.additional_metrics.get("val_dice") is not None else float('-inf')

            if self.scheduler and self.scheduler_step_mode == "epoch":
                self.scheduler.step()

            base_step += steps_this_epoch
            param_group_lrs = {
                f"lr_group_{idx}": float(group["lr"])
                for idx, group in enumerate(self.optimizer.param_groups)
            }
            current_lr = param_group_lrs.get("lr_group_0", max(float(self.optimizer.param_groups[0]["lr"]), 1e-12))

            # RECORD METRICS
            self.metrics_tracker.add_metrics(TrainingMetrics(
                epoch=epoch + 1,
                step=base_step,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                learning_rate=current_lr,
                additional_metrics={
                    "train_dice": avg_train_dice,
                    "val_dice": avg_val_dice,
                    "train_bce_loss": avg_train_bce_loss,
                    "train_dice_loss": avg_train_dice_loss,
                    "val_bce_loss": avg_val_bce_loss,
                    "val_dice_loss": avg_val_dice_loss,
                    **param_group_lrs,
                }
            ))
            
            # LOGGING EPOCH
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={avg_train_loss:.4f}, train_dice={avg_train_dice:.4f}, "
                f"train_bce={avg_train_bce_loss:.4f}, train_dice_loss={avg_train_dice_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_dice={avg_val_dice:.4f}, "
                f"val_bce={avg_val_bce_loss:.4f}, val_dice_loss={avg_val_dice_loss:.4f}"
            )
            
            # SAVE BEST MODEL IF IMPROVED
            if avg_val_dice > prev_best_val:
                self.persister.save_model(self.model, self.cfg.paths.model_path)
                self.logger.info(f"Saved best model at epoch {epoch+1} (dice={avg_val_dice:.4f})")


            # CHECKPOINT
            checkpoint_dir = self.checkpoint_mgr.create_checkpoint(
                epoch=epoch + 1,
                model=self.model,
                optimizer=self.optimizer,
                metrics_tracker=self.metrics_tracker,
                config=self.cfg,
            )

        # WRAP UP BY PERSISTING THE COLLECTED TRAINING HISTORY.
        self.metrics_tracker.end_training()
        self.metrics_tracker.save_to_json(self.cfg.paths.history_path)

        summary = self.metrics_tracker.get_summary_stats() or {}
        duration_minutes = summary.get('duration_minutes')
        total_seconds = duration_minutes * 60 if duration_minutes is not None else None
        best_record = self.metrics_tracker.get_best_metric("val_dice")
        best_metric_value: Optional[float] = None
        best_epoch: Optional[int] = None
        if best_record and best_record.additional_metrics:
            best_metric_value = best_record.additional_metrics.get("val_dice")
            best_epoch = best_record.epoch

        self.logger.log_training_complete(
            total_time=total_seconds,
            best_metric=best_metric_value,
            best_epoch=best_epoch,
        )
        
        # OUTPUT TRAINING CURVES FOR QUICK REVIEW OF PROGRESS.
        plotter = SegmentationPlots(output_directory=self.cfg.paths.run_root, training_history_path=self.cfg.paths.history_path)
        plotter.plot_training_history()
        plotter.plot_learning_rate_schedule()

            
    def _train_epoch(self, train_loader: DataLoader, epoch: int):
        """Run one training pass over the segmentation dataloader."""
        self.model.train()
        loss_sum, dice_sum, steps = 0.0, 0.0, 0
        bce_sum, dice_loss_sum = 0.0, 0.0
        total_elements = 0
        
        # ITERATE OVER BATCHES WITH A PROGRESS BAR
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.training.epochs}")
        for batch in pbar:
            # FETCH IMAGE AND MASK TENSORS FOR THIS MINI-BATCH.
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # ZERO GRADIENTS BEFORE BACKWARD PASS
            self.optimizer.zero_grad()
            
            # ENABLE MIXED PRECISION FOR FASTER FORWARD AND REDUCED MEMORY
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(images)
                bce = self.bce_weight * self.bce_loss(logits, masks)
                dice = self.dice_weight * soft_dice_loss(logits, masks)
                loss = bce + dice
            
            # SCALE LOSS AND BACKWARD FOR STABLE MIXED-PRECISION TRAINING    
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_norm = getattr(self.cfg.training, "grad_clip_norm", 0.0)
            if clip_norm and clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler and self.scheduler_step_mode == "batch":
                self.scheduler.step()

            # UPDATE TRAINING METRICS
            batch_elements = masks.numel()
            loss_sum += loss.item() * batch_elements
            bce_sum += bce.item() * batch_elements
            dice_loss_sum += dice.item() * batch_elements
            total_elements += batch_elements
            dice_sum += dice_coefficient(logits, masks).item()

            steps += 1
            # UPDATE PROGRESS BAR WITH CURRENT LOSS
            avg_dice = dice_sum / max(steps, 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'dice': f'{avg_dice:.3f}',
                'bce': f'{bce.item():.3f}',
                'dice_loss': f'{dice.item():.3f}',
            })
        
        # CALCULATE AVERAGE METRICS    
        avg_train_loss = loss_sum / total_elements if total_elements else float('nan')
        avg_train_dice = dice_sum / max(steps, 1)
        avg_train_bce = bce_sum / total_elements if total_elements else float('nan')
        avg_train_dice_loss = dice_loss_sum / total_elements if total_elements else float('nan')
        
        return avg_train_loss, avg_train_dice, steps, avg_train_bce, avg_train_dice_loss

    @torch.no_grad() # Optimize memory usage and speed up computations
    def _evaluation(self, val_loader: DataLoader):
        """Evaluate segmentation metrics without gradient tracking."""
        
        self.model.eval()
        loss_sum, dice_sum = 0.0, 0.0
        bce_sum, dice_loss_sum = 0.0, 0.0
        total_elements = 0
        steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # RUN VALIDATION FORWARD PASS AND ACCUMULATE METRICS.
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # FORWARD PASS THROUGH SEGMENTATION MODEL
                logits = self.model(images)
                bce = self.bce_weight * self.bce_loss(logits, masks)
                dice = self.dice_weight * soft_dice_loss(logits, masks)
                loss = bce + dice
                
                # ACCUMULATE LOSS AND DICE SCORE
                batch_elements = masks.numel()
                loss_sum += loss.item() * batch_elements
                bce_sum += bce.item() * batch_elements
                dice_loss_sum += dice.item() * batch_elements
                total_elements += batch_elements
                dice_sum += dice_coefficient(logits, masks).item()
                steps += 1
        
        # CALCULATE AVERAGE METRICS ACROSS ALL VALIDATION BATCHES        
        avg_val_loss = loss_sum / total_elements if total_elements else float('nan')
        avg_val_dice = dice_sum / max(steps, 1) if steps else float('nan')
        avg_val_bce = bce_sum / total_elements if total_elements else float('nan')
        avg_val_dice_loss = dice_loss_sum / total_elements if total_elements else float('nan')
        
        return avg_val_loss, avg_val_dice, avg_val_bce, avg_val_dice_loss

    def _log_backbone_trainability(self) -> None:
        """Log how many backbone parameters are trainable to confirm config behaviour."""
        try:
            backbone = getattr(self.model, "backbone")
        except AttributeError:
            return

        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        percentage = (trainable_params / total_params * 100) if total_params else 0.0
        self.logger.info(
            f"Backbone trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)"
        )

        stage_summaries = []
        for name, module in backbone.named_children():
            stage_total = sum(p.numel() for p in module.parameters())
            stage_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            stage_summaries.append(f"{name}: {stage_trainable:,}/{stage_total:,}")
        if stage_summaries:
            breakdown = ", ".join(stage_summaries)
            self.logger.info(f"Backbone stage breakdown -> {breakdown}")

    def _log_model_summary(self) -> None:
        """Log high-level architecture details and parameter counts."""
        try:
            arch_lines = str(self.model).splitlines()
            if arch_lines:
                header = ["Model architecture (top level):"]
                for line in arch_lines[:40]:  # trim extremely long dumps
                    header.append(f"  {line}")
                if len(arch_lines) > 40:
                    header.append("  ... (truncated)")
                self.logger.info("\n".join(header))
        except Exception as exc:
            self.logger.warning(f"Failed to stringify model architecture: {exc}")

        total_params = 0
        trainable_params = 0
        component_stats: Dict[str, Dict[str, int]] = {}
        for name, param in self.model.named_parameters():
            count = param.numel()
            total_params += count
            if param.requires_grad:
                trainable_params += count
            prefix = name.split(".", 1)[0]
            stats = component_stats.setdefault(prefix, {"total": 0, "trainable": 0})
            stats["total"] += count
            if param.requires_grad:
                stats["trainable"] += count

        frozen_params = total_params - trainable_params
        self.logger.info(
            f"Parameter summary -> total: {total_params/1e6:.2f}M | "
            f"trainable: {trainable_params/1e6:.2f}M | frozen: {frozen_params/1e6:.2f}M"
        )
        for prefix, stats in sorted(component_stats.items()):
            self.logger.info(
                f"  {prefix:<12} :: total={stats['total']/1e6:.3f}M | "
                f"trainable={stats['trainable']/1e6:.3f}M | frozen={(stats['total']-stats['trainable'])/1e6:.3f}M"
            )

    def _log_forward_shape_snapshot(self, loader: DataLoader) -> None:
        """Log tensor shapes through the backbone and decoder for a single batch."""
        try:
            iterator = iter(loader)
            batch = next(iterator)
        except StopIteration:
            self.logger.warning("Unable to log shape snapshot: training loader is empty")
            return
        except Exception as exc:
            self.logger.warning(f"Unable to create loader iterator for shape snapshot: {exc}")
            return

        images = batch.get("image")
        if images is None:
            self.logger.warning("Shape snapshot skipped: batch lacks 'image' tensor")
            return

        with torch.no_grad():
            try:
                sample = images[:1].to(self.device)
                skips, deep = self.model.backbone(sample)
                self.logger.info("Forward shape snapshot (single sample):")
                for idx, skip in enumerate(skips):
                    self.logger.info(f"  skip{idx+1}: {tuple(skip.shape)}")

                h = self.model.bottleneck(deep)
                self.logger.info(f"  bottleneck: {tuple(h.shape)}")

                h = self.model.dec4(h, skips[3])
                self.logger.info(f"  dec4: {tuple(h.shape)}")

                h = self.model.dec3(h, skips[2])
                self.logger.info(f"  dec3: {tuple(h.shape)}")

                h = self.model.dec2(h, skips[1])
                self.logger.info(f"  dec2: {tuple(h.shape)}")

                h = self.model.dec1(h, skips[0])
                self.logger.info(f"  dec1: {tuple(h.shape)}")

                logits = self.model.head(h)
                self.logger.info(f"  head logits: {tuple(logits.shape)}")
            except Exception as exc:
                self.logger.warning(f"Failed to log forward shape snapshot: {exc}")
    
    def _load_data(self):
        """Prepare train/validation datasets and wrap them in DataLoaders."""
        manager = SIDDatasetManager(
            dataset_name=self.cfg.data.dataset_name,
            use_streaming=self.cfg.data.use_streaming,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )

        train_joint_tf, train_image_tf, train_mask_tf = build_segmentation_transforms(
            self.cfg.data.image_size,
            self.cfg.model.normalize_mean,
            self.cfg.model.normalize_std,
            self.cfg.data.augment,
            is_train=True,
        )
        val_joint_tf, val_image_tf, val_mask_tf = build_segmentation_transforms(
            self.cfg.data.image_size,
            self.cfg.model.normalize_mean,
            self.cfg.model.normalize_std,
            self.cfg.data.augment,
            is_train=False,
        )

        train_ds = manager.get_split(
            split_type=TRAIN,
            max_samples=self.cfg.data.train_samples,
            filter_fn=DatasetFilters.tampered_with_masks
        )
        self._save_augmentation_preview(train_ds, train_joint_tf, train_image_tf, train_mask_tf)
        val_ds = manager.get_split(
            split_type=VALIDATION,
            max_samples=self.cfg.data.val_samples,
            filter_fn=DatasetFilters.tampered_with_masks
        )
        num_workers = self.cfg.loader.num_workers if not self.cfg.data.use_streaming else 0
        loader_common_kwargs = {
            "batch_size": self.cfg.loader.batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_common_kwargs["prefetch_factor"] = self.cfg.loader.prefetch_factor
            loader_common_kwargs["persistent_workers"] = self.cfg.loader.persistent_workers

        train_loader = DataLoader(
            SIDClassificationDataset(train_ds, image_size=self.cfg.data.image_size,
                                     normalize_mean=self.cfg.model.normalize_mean,
                                     normalize_std=self.cfg.model.normalize_std,
                                     transform=train_image_tf,
                                     transform_mask=train_mask_tf,
                                     joint_transform=train_joint_tf,
                                     return_mask=True),
            shuffle=not self.cfg.data.use_streaming and self.cfg.loader.shuffle_train,
            **loader_common_kwargs,
        )
        val_loader = DataLoader(
            SIDClassificationDataset(val_ds, image_size=self.cfg.data.image_size,
                                     normalize_mean=self.cfg.model.normalize_mean,
                                     normalize_std=self.cfg.model.normalize_std,
                                     transform=val_image_tf,
                                     transform_mask=val_mask_tf,
                                     joint_transform=val_joint_tf,
                                     return_mask=True),
            shuffle=False,
            **loader_common_kwargs,
        )
        return train_loader, val_loader

    def _save_augmentation_preview(self, dataset, joint_transform, image_transform, mask_transform):
        """Optionally save a grid of augmented image-mask pairs for quick sanity checks."""
        augment_cfg = getattr(self.cfg.data, "augment", None)
        if not augment_cfg or not getattr(augment_cfg, "enable", False):
            return

        preview_samples = getattr(augment_cfg, "preview_samples", 0)
        if preview_samples <= 0:
            return

        try:
            dataset_length = len(dataset)
        except TypeError:
            self.logger.warning("Skipping augmentation preview: dataset length unavailable")
            return

        if dataset_length == 0:
            self.logger.warning("Skipping augmentation preview: dataset is empty")
            return

        count = min(preview_samples, dataset_length)
        rng = torch.Generator().manual_seed(getattr(augment_cfg, "preview_seed", 1234))
        indices = torch.randperm(dataset_length, generator=rng)[:count]

        from torchvision.utils import make_grid, save_image
        tile_images: List[torch.Tensor] = []
        mean = torch.tensor(self.cfg.model.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.cfg.model.normalize_std).view(3, 1, 1)

        processed = 0
        for idx in indices:
            example = dataset[int(idx)]
            image = SIDClassificationDataset.to_rgb(example["image"])
            mask = example.get("mask")
            if mask is None:
                continue

            augmented_image, augmented_mask = joint_transform(image, mask)
            image_tensor = image_transform(augmented_image)
            mask_tensor = mask_transform(augmented_mask)

            denorm = (image_tensor * std + mean).clamp(0.0, 1.0)
            overlay = self._apply_mask_overlay(denorm, mask_tensor)
            mask_rgb = torch.zeros_like(denorm)
            mask_rgb[0] = mask_tensor.squeeze(0)

            tile_images.extend([denorm, overlay, mask_rgb])
            processed += 1

        if not tile_images:
            self.logger.warning("Skipping augmentation preview: no samples processed successfully")
            return

        tiles = torch.stack(tile_images)
        grid = make_grid(tiles, nrow=3, padding=4, pad_value=0.1)
        preview_path = self.cfg.paths.run_root / "augmentation_preview_segmentation.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, preview_path)
        self.logger.info(
            f"Saved segmentation augmentation preview with {processed} samples "
            f"({len(tile_images)} panels) to {preview_path}"
        )

    @staticmethod
    def _apply_mask_overlay(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, color=(1.0, 0.0, 0.0), alpha: float = 0.4):
        """Blend a colored mask onto the denormalised image for visual inspection."""
        color_tensor = torch.tensor(color, device=image_tensor.device, dtype=image_tensor.dtype).view(3, 1, 1)
        alpha_tensor = alpha * mask_tensor
        overlay = (1 - alpha_tensor) * image_tensor + alpha_tensor * color_tensor
        return overlay.clamp(0.0, 1.0)
