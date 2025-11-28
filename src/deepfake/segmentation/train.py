from dataclasses import asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel

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
from deepfake.utils.losses import FocalLoss

try:
    from torchinfo import summary as torchinfo_summary  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torchinfo_summary = None  # type: ignore

class Trainer:
    """
    CLASS-BASED TRAINER FOR TAMPER SEGMENTATION WITH MIXED PRECISION,
    METRICS TRACKING, CHECKPOINTING, AND PLOTTING.
    """
    # ORGANISES DATA, MODEL, LOSSES, LOGGING, AND CHECKPOINTS ACROSS THE SEGMENTATION WORKFLOW.

    def __init__(
        self,
        cfg: Config,
        logger: SidLogger,
        metrics_tracker: TrainingMetricsTracker,
        checkpoint_mgr: CheckpointManager,
        persister: IModelPersister = None,
        verbose: bool = True
    ):
        self.cfg = cfg
        # STORE REFERENCES TO LOGGER, METRICS TRACKER, AND CHECKPOINT MANAGER FOR LATER USE.
        self.logger = logger
        self.device = torch.device(cfg.training.device)
        self.verbose = verbose

        # INITIALISE THE SEGMENTATION MODEL PLUS OPTIMISER, LOSSES, AND MIXED PRECISION HELPERS.
        self.model = TamperSegmentationModel(model_cfg=cfg.model, in_channels=3, out_channels=1).to(self.device)
        self._log_model_summary()
        self._save_model_visualizations()
        loss_cfg = getattr(cfg.training, "loss", None)
        loss_type = getattr(loss_cfg, "type", "bce") if loss_cfg is not None else "bce"
        loss_type = str(loss_type).lower()
        self.primary_loss_name = "focal" if loss_type == "focal" else "bce"
        if self.primary_loss_name == "focal":
            alpha = getattr(loss_cfg, "focal_alpha", 0.25)
            gamma = getattr(loss_cfg, "focal_gamma", 2.0)
            self.primary_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        else:
            self.primary_loss_fn = nn.BCEWithLogitsLoss()

        self.primary_weight = getattr(loss_cfg, "bce_weight", 0.5) if loss_cfg is not None else 0.5
        self.dice_weight = getattr(loss_cfg, "dice_weight", 0.5) if loss_cfg is not None else 0.5
        total_weight = self.primary_weight + self.dice_weight
        if total_weight > 0:
            self.primary_weight /= total_weight
            self.dice_weight /= total_weight
        self.optimizer = OptimizerFactory(self.model.parameters(), cfg.training)
        self.amp_enabled = (self.device.type == "cuda" and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.scheduler = None
        self.scheduler_step_mode = "epoch"
        self.grad_clip_threshold = getattr(cfg.training, "grad_clip_norm", 0.0)
        self.ema_model: Optional[AveragedModel] = None
        ema_decay = getattr(cfg.training, "ema_decay", 0.0)
        if ema_decay and ema_decay > 0:
            self.ema_model = AveragedModel(
                self.model,
                avg_fn=lambda averaged_param, model_param, _: ema_decay * averaged_param + (1.0 - ema_decay) * model_param,
            )
            self.ema_model.to(self.device)

        # SAVE HANDLES TO PERSISTENCE AND TRACKING UTILITIES SO RUNS CAN RESUME CLEANLY.
        self.persister = persister or TorchModelPersister()
        self.metrics_tracker = metrics_tracker
        self.checkpoint_mgr = checkpoint_mgr

        primary = self.primary_loss_name
        metric_directions = {
            "train_dice": "max",
            "val_dice": "max",
            f"train_{primary}_weighted": "min",
            "train_dice_weighted": "min",
            f"val_{primary}_weighted": "min",
            "val_dice_weighted": "min",
            f"train_{primary}_raw": "min",
            "train_dice_raw": "min",
            f"val_{primary}_raw": "min",
            "val_dice_raw": "min",
            "grad_norm_avg": "last",
            "grad_norm_max": "last",
            "grad_clip_frac": "min",
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

        if self.cfg.paths.model_path.is_file():
            self.persister.load_model(self.model, self.cfg.paths.model_path, device=self.device)
            if self.ema_model is not None:
                self.ema_model.module.load_state_dict(self.model.state_dict())
            self.logger.info(f"Loaded best model from {self.cfg.paths.model_path}")

    def train(self, start_epoch: int = 0):
        """Train the tamper segmentation model on the configured SID subsets."""
        prev_best = None
        prev_best_val = float('-inf')
        
        # PREPARE TRAIN AND VALIDATION LOADERS LIMITED TO TAMPERED SAMPLES THAT INCLUDE MASKS.
        train_loader, val_loader = self._load_data()
        if self.verbose:
            self._log_forward_shape_snapshot(train_loader)

        try:
            steps_per_epoch = len(train_loader)
        except (TypeError, AttributeError):
            steps_per_epoch = 0
        # BUILD THE LR SCHEDULER ONLY AFTER WE KNOW HOW MANY BATCHES ARE IN AN EPOCH.
        self.scheduler, self.scheduler_step_mode = SchedulerFactory.create(
            self.optimizer,
            self.cfg.training,
            steps_per_epoch=steps_per_epoch,
        )
        if self.scheduler and self.scheduler_step_mode == "epoch" and start_epoch > 0:
            self.scheduler.last_epoch = start_epoch - 1

        # START METRIC TRACKING SO WE CAPTURE TIMESTAMPS AND HISTORY FROM THE FIRST EPOCH.
        self.metrics_tracker.start_training()
        base_step = self.metrics_tracker.metrics[-1].step if self.metrics_tracker.metrics else 0

        # MAIN LOOP: TRAIN AN EPOCH, VALIDATE IT, LOG RESULTS, AND HANDLE CHECKPOINTS.
        for epoch in range(start_epoch, self.cfg.training.epochs):
            (
                avg_train_loss,
                avg_train_dice,
                steps_this_epoch,
                avg_train_primary_weighted,
                avg_train_dice_weighted,
                avg_train_primary_raw,
                avg_train_dice_raw,
                avg_grad_norm,
                max_grad_norm,
                grad_clip_fraction,
            ) = self._train_epoch(train_loader, epoch)
            (
                avg_val_loss,
                avg_val_dice,
                avg_val_primary_weighted,
                avg_val_dice_weighted,
                avg_val_primary_raw,
                avg_val_dice_raw,
            ) = self._evaluation(val_loader)
            
            # CHECK THE PREVIOUS BEST VALIDATION DICE SO WE KNOW IF THIS EPOCH IMPROVES IT.
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

            # RECORD THE FULL SET OF TRAINING AND VALIDATION METRICS FOR THIS EPOCH.
            additional_metrics = {
                "train_dice": avg_train_dice,
                "val_dice": avg_val_dice,
                f"train_{self.primary_loss_name}_weighted": avg_train_primary_weighted,
                "train_dice_weighted": avg_train_dice_weighted,
                f"val_{self.primary_loss_name}_weighted": avg_val_primary_weighted,
                "val_dice_weighted": avg_val_dice_weighted,
                f"train_{self.primary_loss_name}_raw": avg_train_primary_raw,
                "train_dice_raw": avg_train_dice_raw,
                f"val_{self.primary_loss_name}_raw": avg_val_primary_raw,
                "val_dice_raw": avg_val_dice_raw,
                "grad_norm_avg": avg_grad_norm,
                "grad_norm_max": max_grad_norm,
                "grad_clip_frac": grad_clip_fraction,
                **param_group_lrs,
            }

            self.metrics_tracker.add_metrics(TrainingMetrics(
                epoch=epoch + 1,
                step=base_step,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                learning_rate=current_lr,
                additional_metrics=additional_metrics
            ))
            
            # LOG A FRIENDLY SUMMARY SO USERS CAN FOLLOW PROGRESS.
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={avg_train_loss:.4f}, train_dice={avg_train_dice:.4f}, "
                f"train_{self.primary_loss_name}_w={avg_train_primary_weighted:.4f}, train_dice_w={avg_train_dice_weighted:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_dice={avg_val_dice:.4f}, "
                f"val_{self.primary_loss_name}_w={avg_val_primary_weighted:.4f}, val_dice_w={avg_val_dice_weighted:.4f}, "
                f"grad_avg={avg_grad_norm:.4f}, grad_max={max_grad_norm:.4f}, clip_frac={grad_clip_fraction:.1%}"
            )
            
            # SAVE THE MODEL WHEN IT ACHIEVES A BETTER VALIDATION DICE THAN ANY PRIOR EPOCH.
            if avg_val_dice > prev_best_val:
                self.persister.save_model(self._model_for_export(), self.cfg.paths.model_path)
                self.logger.info(f"Saved best model at epoch {epoch+1} (dice={avg_val_dice:.4f})")


            # CREATE SCHEDULED CHECKPOINTS SO TRAINING CAN RECOVER AFTER INTERRUPTIONS.
            checkpoint_dir = self.checkpoint_mgr.create_checkpoint(
                epoch=epoch + 1,
                model=self._model_for_export(),
                optimizer=self.optimizer,
                metrics_tracker=self.metrics_tracker,
                config=self.cfg,
            )
            if checkpoint_dir and self.ema_model is not None:
                torch.save(self.ema_model.state_dict(), checkpoint_dir / "ema_state.pth")

        # AFTER TRAINING FINISHES, FINALISE AND SAVE METRICS FOR LATER ANALYSIS.
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
        
        # RENDER TRAINING AND LR PLOTS SO USERS CAN REVIEW PROGRESS QUICKLY.
        plotter = SegmentationPlots(output_directory=self.cfg.paths.run_root, training_history_path=self.cfg.paths.history_path)
        plotter.plot_training_history()
        plotter.plot_learning_rate_schedule()

            
    def _train_epoch(self, train_loader: DataLoader, epoch: int):
        """RUN ONE FULL TRAINING EPOCH OVER THE SEGMENTATION DATA LOADER."""
        self.model.train()
        loss_sum, dice_sum, steps = 0.0, 0.0, 0
        primary_weighted_sum, dice_weighted_sum = 0.0, 0.0
        primary_raw_sum, dice_raw_sum = 0.0, 0.0
        total_elements = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_clip_events = 0
        clip_threshold = self.grad_clip_threshold
        
        # LOOP OVER BATCHES WITH A PROGRESS BAR FOR USER FEEDBACK.
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.cfg.training.epochs}",
            disable=not self.verbose,
        )
        for batch in pbar:
            # FETCH IMAGE AND MASK TENSORS FOR THIS MINI-BATCH.
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # RESET OPTIMISER GRADIENTS BEFORE BACKPROPAGATION.
            self.optimizer.zero_grad()
            
            # ENABLE MIXED PRECISION TO SPEED UP TRAINING AND REDUCE MEMORY FOOTPRINT.
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(images)
                primary_raw = self.primary_loss_fn(logits, masks)
                dice_raw = soft_dice_loss(logits, masks)
                primary_term = self.primary_weight * primary_raw
                dice_term = self.dice_weight * dice_raw
                loss = primary_term + dice_term
            
            # SCALE THE LOSS FOR MIXED PRECISION, BACKPROPAGATE, AND STEP THE OPTIMISER.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if clip_threshold and clip_threshold > 0:
                grad_norm = float(clip_grad_norm_(self.model.parameters(), clip_threshold))
                if grad_norm > clip_threshold:
                    grad_clip_events += 1
            else:
                grad_norm = self._compute_grad_norm(self.model.parameters())
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler and self.scheduler_step_mode == "batch":
                self.scheduler.step()
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            # ACCUMULATE PER-BATCH LOSS CONTRIBUTIONS AND DICE OVER THE WHOLE DATASET.
            batch_elements = masks.numel()
            loss_sum += loss.item() * batch_elements
            primary_weighted_sum += primary_term.item() * batch_elements
            dice_weighted_sum += dice_term.item() * batch_elements
            primary_raw_sum += primary_raw.item() * batch_elements
            dice_raw_sum += dice_raw.item() * batch_elements
            total_elements += batch_elements
            dice_sum += dice_coefficient(logits, masks).item()

            steps += 1
            # SHOW CURRENT LOSS COMPONENTS ON THE PROGRESS BAR FOR QUICK MONITORING.
            avg_dice = dice_sum / max(steps, 1)
            if self.verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'dice': f'{avg_dice:.3f}',
                    f'{self.primary_loss_name}_w': f'{primary_term.item():.3f}',
                    'dice_w': f'{dice_term.item():.3f}',
                })
        
        # CALCULATE AVERAGE METRICS TO LOG AND SAVE FOR THIS EPOCH.
        avg_train_loss = loss_sum / total_elements if total_elements else float('nan')
        avg_train_dice = dice_sum / max(steps, 1)
        avg_train_primary_weighted = primary_weighted_sum / total_elements if total_elements else float('nan')
        avg_train_dice_weighted = dice_weighted_sum / total_elements if total_elements else float('nan')
        avg_train_primary_raw = primary_raw_sum / total_elements if total_elements else float('nan')
        avg_train_dice_raw = dice_raw_sum / total_elements if total_elements else float('nan')
        step_count = max(steps, 1)
        avg_grad_norm = grad_norm_sum / step_count
        grad_clip_fraction = grad_clip_events / step_count
        if clip_threshold and clip_threshold > 0 and grad_clip_fraction > 0.3 and self.verbose:
            self.logger.logger.warning(
                f"Gradient clipping triggered on {grad_clip_fraction:.1%} of batches (threshold {clip_threshold})."
            )

        return (
            avg_train_loss,
            avg_train_dice,
            steps,
            avg_train_primary_weighted,
            avg_train_dice_weighted,
            avg_train_primary_raw,
            avg_train_dice_raw,
            avg_grad_norm,
            grad_norm_max,
            grad_clip_fraction,
        )

    @torch.no_grad()  # DISABLE GRADIENTS DURING VALIDATION TO SAVE MEMORY AND TIME.
    def _evaluation(self, val_loader: DataLoader):
        """EVALUATE SEGMENTATION METRICS WITHOUT UPDATING MODEL WEIGHTS."""
        
        self.model.eval()
        loss_sum, dice_sum = 0.0, 0.0
        primary_weighted_sum, dice_weighted_sum = 0.0, 0.0
        primary_raw_sum, dice_raw_sum = 0.0, 0.0
        total_elements = 0
        steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # RUN THE VALIDATION FORWARD PASS AND ACCUMULATE METRICS.
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # FORWARD PASS THROUGH SEGMENTATION MODEL
                logits = self.model(images)
                primary_raw = self.primary_loss_fn(logits, masks)
                dice_raw = soft_dice_loss(logits, masks)
                primary_term = self.primary_weight * primary_raw
                dice_term = self.dice_weight * dice_raw
                loss = primary_term + dice_term
                
                # ACCUMULATE LOSS AND DICE SCORE
                batch_elements = masks.numel()
                loss_sum += loss.item() * batch_elements
                primary_weighted_sum += primary_term.item() * batch_elements
                dice_weighted_sum += dice_term.item() * batch_elements
                primary_raw_sum += primary_raw.item() * batch_elements
                dice_raw_sum += dice_raw.item() * batch_elements
                total_elements += batch_elements
                dice_sum += dice_coefficient(logits, masks).item()
                steps += 1
        
        # COMPUTE AVERAGE VALIDATION METRICS SO WE CAN COMPARE EPOCHS FAIRLY.
        avg_val_loss = loss_sum / total_elements if total_elements else float('nan')
        avg_val_dice = dice_sum / max(steps, 1) if steps else float('nan')
        avg_val_primary_weighted = primary_weighted_sum / total_elements if total_elements else float('nan')
        avg_val_dice_weighted = dice_weighted_sum / total_elements if total_elements else float('nan')
        avg_val_primary_raw = primary_raw_sum / total_elements if total_elements else float('nan')
        avg_val_dice_raw = dice_raw_sum / total_elements if total_elements else float('nan')

        return (
            avg_val_loss,
            avg_val_dice,
            avg_val_primary_weighted,
            avg_val_dice_weighted,
            avg_val_primary_raw,
            avg_val_dice_raw,
        )

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
        def _fmt_param_count(count: int) -> str:
            if count >= 1_000_000:
                return f"{count/1_000_000:.2f}M"
            if count >= 1_000:
                return f"{count/1_000:.2f}K"
            return f"{count}"

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
            "Parameter summary -> "
            f"total: {_fmt_param_count(total_params)} ({total_params}) | "
            f"trainable: {_fmt_param_count(trainable_params)} ({trainable_params}) | "
            f"frozen: {_fmt_param_count(frozen_params)} ({frozen_params})"
        )
        for prefix, stats in sorted(component_stats.items()):
            frozen = stats['total'] - stats['trainable']
            self.logger.info(
                f"  {prefix:<12} :: total={_fmt_param_count(stats['total'])} ({stats['total']}) | "
                f"trainable={_fmt_param_count(stats['trainable'])} ({stats['trainable']}) | "
                f"frozen={_fmt_param_count(frozen)} ({frozen})"
            )

    def _save_model_visualizations(self) -> None:
        """Persist torchinfo summary and torchviz graph into the run directory."""
        run_root = Path(self.cfg.paths.run_root)
        run_root.mkdir(parents=True, exist_ok=True)

        was_training = self.model.training
        self.model.eval()
        # TORCHINFO SUMMARY PROVIDES A TEXT OVERVIEW OF LAYERS, SHAPES, AND PARAM COUNTS.
        if torchinfo_summary is not None:
            try:
                input_size = (1, 3, self.cfg.data.image_size, self.cfg.data.image_size)
                summary_text = str(
                    torchinfo_summary(
                        self.model,
                        input_size=input_size,
                        device=str(self.device),
                        verbose=0,
                        col_names=("input_size", "output_size", "num_params", "mult_adds"),
                    )
                )
                summary_path = run_root / "model_summary.txt"
                summary_path.write_text(summary_text)
                self.logger.info(f"Wrote torchinfo summary to {summary_path}")
            except Exception as exc:  # pragma: no cover - diagnostic
                self.logger.warning(f"Failed to save torchinfo summary: {exc}")
        else:
            self.logger.debug("torchinfo not installed; skipping textual model summary.")
        self.model.train(was_training)

    def _model_for_export(self) -> nn.Module:
        """Return EMA weights when available so saved checkpoints are smoothed."""
        return self.ema_model.module if self.ema_model is not None else self.model

    def load_ema_state(self, ema_path: Path) -> None:
        """Restore EMA weights from disk when resuming from a checkpoint."""
        if self.ema_model is None or not ema_path.exists():
            return
        state_dict = torch.load(ema_path, map_location=self.device)
        self.ema_model.load_state_dict(state_dict)
        self.ema_model.to(self.device)

    @staticmethod
    def _compute_grad_norm(parameters) -> float:
        total_sq = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            param_norm = p.grad.data.float().norm(2)
            total_sq += float(param_norm.item() ** 2)
        return math.sqrt(total_sq) if total_sq > 0 else 0.0

    def _log_forward_shape_snapshot(self, loader: DataLoader) -> None:
        """Log tensor shapes through the backbone and decoder for a single batch."""
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            self.logger.warning("Shape snapshot skipped: loader has no dataset attribute")
            return
        # SKIP ITERABLE DATASETS BECAUSE THEY CANNOT BE SAFELY INDEXED FOR A SAMPLE.
        from torch.utils.data import IterableDataset  # local import to keep optional dependency light

        if isinstance(dataset, IterableDataset):
            self.logger.debug("Shape snapshot skipped: dataset is iterable-only")
            return

        try:
            if len(dataset) == 0:  # type: ignore[arg-type]
                self.logger.warning("Shape snapshot skipped: dataset is empty")
                return
        except TypeError:
            self.logger.debug("Shape snapshot skipped: dataset length unavailable")
            return

        try:
            sample_item = dataset[0]
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to index dataset for shape snapshot: {exc}")
            return

        image_tensor = sample_item.get("image")
        if image_tensor is None:
            self.logger.warning("Shape snapshot skipped: dataset sample lacks 'image'")
            return
        if not isinstance(image_tensor, torch.Tensor):
            try:
                image_tensor = torch.as_tensor(image_tensor)
            except Exception as exc:
                self.logger.warning(f"Shape snapshot skipped: unable to convert image to tensor ({exc})")
                return

        sample = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
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
        """OPTIONALLY SAVE A GRID OF AUGMENTED IMAGE-MASK PAIRS FOR QUICK SANITY CHECKS."""
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
        """BLEND A COLOURED MASK ONTO THE DENORMALISED IMAGE SO TAMPER REGIONS ARE EASY TO SEE."""
        color_tensor = torch.tensor(color, device=image_tensor.device, dtype=image_tensor.dtype).view(3, 1, 1)
        alpha_tensor = alpha * mask_tensor
        overlay = (1 - alpha_tensor) * image_tensor + alpha_tensor * color_tensor
        return overlay.clamp(0.0, 1.0)
