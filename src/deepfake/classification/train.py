import os
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode
from pathlib import Path
from dataclasses import asdict
from torch.amp import autocast, GradScaler
import numpy as np
import math
import torchvision.utils as vutils
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel

from deepfake.config import Config
from deepfake.utils.optimizer_factory import OptimizerFactory
from deepfake.utils.model_persister import TorchModelPersister
from deepfake.data.dataset_manager import SIDDatasetManager, TRAIN, VALIDATION
from deepfake.utils.logger import SidLogger as SidLogger
from deepfake.utils.model_persister import TorchModelPersister, IModelPersister
from deepfake.utils.training_metrics_tracker import TrainingMetrics, TrainingMetricsTracker
from deepfake.utils.scheduler_factory import SchedulerFactory
from deepfake.utils.augmentation_factory import build_classification_transform
from deepfake.visualization.classification_plots import ClassificationPlots
from deepfake.data.dataset import SIDClassificationDataset
from deepfake.utils.checkpoint_manager import CheckpointManager
from .model import BaselineClassifier


class Trainer:
    def __init__(
        self,
        cfg: Config,
        logger: SidLogger,
        metrics_tracker: TrainingMetricsTracker,
        checkpoint_mgr: CheckpointManager,
        persister: IModelPersister = None
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        # Bookkeeping surfaces: logs, metrics, checkpoints
        self.logger = logger
        self.metrics_tracker = metrics_tracker
        self.checkpoint_mgr = checkpoint_mgr
        self.persister = persister or TorchModelPersister()
        
        
        # Model, optimizer, criterion, scaler
        # Core optimisation components
        self.model = BaselineClassifier(num_classes=cfg.model.num_classes).to(self.device)
        self.optimizer = OptimizerFactory(self.model.parameters(), cfg.training)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(cfg.training, "label_smoothing", 0.0))
        self.amp_enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.scheduler = None
        self.scheduler_step_mode = "epoch"
        self.ema_model = None

        if cfg.training.ema_decay > 0:
            decay = cfg.training.ema_decay
            self.ema_model = AveragedModel(
                self.model,
                avg_fn=lambda averaged_param, model_param, _: decay * averaged_param + (1.0 - decay) * model_param,
            )
            self.ema_model.to(self.device)

        # Load best model if exists
        if os.path.isfile(cfg.paths.model_path):
            self.persister.load_model(self.model, cfg.paths.model_path, device=self.device)
            self.logger.info(f"Loaded best model from {cfg.paths.model_path}")       
        
        self.logger.log_training_config(asdict(cfg))

    def train(self, start_epoch: int = 0):
        """Train the lightweight classifier on the configured SID subsets."""
        prev_best = None
        prev_best_val = float('-inf')
        
        # LOAD THE DATA
        train_loader, val_loader = self._load_data(self.cfg)

        try:
            steps_per_epoch = len(train_loader)
        except (TypeError, AttributeError):
            steps_per_epoch = 0
        # Build scheduler (if configured) now that we know steps/epoch
        self.scheduler, self.scheduler_step_mode = SchedulerFactory.create(
            self.optimizer,
            self.cfg.training,
            steps_per_epoch=steps_per_epoch,
        )
        if self.scheduler and self.scheduler_step_mode == "epoch" and start_epoch > 0:
            self.scheduler.last_epoch = start_epoch - 1

        # START TRAINING
        self.metrics_tracker.start_training()
        base_step = self.metrics_tracker.metrics[-1].step if self.metrics_tracker.metrics else 0

        # Main training loop: fit, validate, log, checkpoint
        for epoch in range(start_epoch, self.cfg.training.epochs):
            train_acc, avg_train_loss, batch_count = self._train_epoch(train_loader, epoch)
            eval_model = self.ema_model.module if self.ema_model is not None else self.model
            val_acc, avg_val_loss = self._evaluation(val_loader, model=eval_model)

            prev_best = self.metrics_tracker.get_best_metric("val_acc")
            prev_best_val = prev_best.val_acc if prev_best and prev_best.val_acc is not None else float('-inf')

            if self.scheduler and self.scheduler_step_mode == "epoch":
                self.scheduler.step()

            base_step += batch_count
            current_lr = max(float(self.optimizer.param_groups[0]['lr']), 1e-12)

            self.metrics_tracker.add_metrics(TrainingMetrics(
                epoch=epoch + 1,
                step=base_step,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=current_lr,
            ))

            train_acc_display = (
                train_acc if (train_acc is not None and not math.isnan(train_acc)) else None
            )
            val_acc_display = val_acc if val_acc is not None and not math.isnan(val_acc) else None
            self.logger.log_epoch_results(
                epoch,
                train_loss=avg_train_loss,
                train_acc=train_acc_display,
                val_loss=avg_val_loss,
                val_acc=val_acc_display,
                lr=current_lr,
            )

            if val_acc is not None and not math.isnan(val_acc) and val_acc > prev_best_val:
                self.persister.save_model(self._model_for_export(), self.cfg.paths.model_path)
                self.logger.info(f"Saved best model at epoch {epoch + 1} (val_acc={val_acc:.4f})")

            checkpoint_dir = self.checkpoint_mgr.create_checkpoint(
                epoch=epoch + 1,
                model=self._model_for_export(),
                optimizer=self.optimizer,
                metrics_tracker=self.metrics_tracker,
                config=self.cfg,
            )
            if checkpoint_dir and self.ema_model is not None:
                torch.save(self.ema_model.state_dict(), checkpoint_dir / "ema_state.pth")

        # FINISH
        self.metrics_tracker.end_training()
        self.metrics_tracker.save_to_json(self.cfg.paths.history_path)

        summary = self.metrics_tracker.get_summary_stats() or {}
        duration_minutes = summary.get('duration_minutes')
        total_seconds = duration_minutes * 60 if duration_minutes is not None else None
        best_record = self.metrics_tracker.get_best_metric("val_acc")
        best_metric_value = best_record.val_acc if best_record and best_record.val_acc is not None else None
        best_epoch = best_record.epoch if best_record else None

        self.logger.log_training_complete(
            total_time=total_seconds,
            best_metric=best_metric_value,
            best_epoch=best_epoch,
        )

        # PLOT
        plotter = ClassificationPlots(output_directory=self.cfg.paths.run_root, training_history_path=self.cfg.paths.history_path)
        plotter.plot_training_history()
        plotter.plot_learning_rate_schedule()
        

    def _train_epoch(self, train_loader: DataLoader, epoch: int = 0):
        """Run one training epoch and report metrics plus batch count."""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.training.epochs}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_count += 1

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_norm = getattr(self.cfg.training, "grad_clip_norm", 0.0)
            if clip_norm and clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            if self.scheduler and self.scheduler_step_mode == "batch":
                self.scheduler.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            train_total += batch_size
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        train_acc = train_correct / train_total if train_total else None
        avg_train_loss = train_loss / train_total if train_total else float('nan')

        return train_acc, avg_train_loss, batch_count
    
    @torch.no_grad() # Optimize memory usage and speed up computations
    def _evaluation(self, val_loader: DataLoader, model: Optional[nn.Module] = None):
        """Evaluate model (EMA if provided) without gradient tracking."""
        eval_model = model if model is not None else self.model
        eval_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # FORWARD PASS THROUGH CLASSIFICATION MODEL
                outputs = eval_model(images)
                loss = self.criterion(outputs, labels)
                
                # ACCUMULATE LOSS
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                
                # COUNT TOTAL SAMPLES AND CORRECT PREDICTIONS
                _, predicted = torch.max(outputs, 1)
                val_total += batch_size
                val_correct += (predicted == labels).sum().item()

        # CALCULATE ACCURACY AND AVERAGE LOSS WITH SAFE DIVISION
        val_acc = val_correct / val_total if val_total > 0 else None
        avg_val_loss = (val_loss / val_total if val_total > 0 else float('nan'))

        return val_acc, avg_val_loss
    
    
        
    def _save_augmentation_preview(self, dataset, transform):
        """Optionally save a grid of augmented training samples."""
        augment_cfg = getattr(self.cfg.data, 'augment', None)
        if not augment_cfg or not getattr(augment_cfg, 'enable', False):
            return

        preview_samples = getattr(augment_cfg, 'preview_samples', 0)
        if preview_samples <= 0:
            return

        try:
            dataset_length = len(dataset)
        except TypeError:
            self.logger.warning('Skipping augmentation preview: dataset length unavailable')
            return

        if dataset_length == 0:
            self.logger.warning('Skipping augmentation preview: dataset is empty')
            return

        count = min(preview_samples, dataset_length)
        rng = np.random.default_rng(getattr(augment_cfg, 'preview_seed', 1234))
        indices = rng.integers(0, dataset_length, size=count)

        augmented_images = []
        for idx in indices:
            try:
                example = dataset[int(idx)]
            except Exception as exc:
                self.logger.warning(f'Failed to fetch sample {idx} for augmentation preview: {exc}')
                continue

            raw_image = SIDClassificationDataset.to_rgb(example['image'])
            augmented = transform(raw_image)
            augmented_images.append(augmented)

        if not augmented_images:
            self.logger.warning('Skipping augmentation preview: no samples were processed successfully')
            return

        device = augmented_images[0].device
        dtype = augmented_images[0].dtype
        mean = torch.tensor(self.cfg.model.normalize_mean, dtype=dtype, device=device).view(3, 1, 1)
        std = torch.tensor(self.cfg.model.normalize_std, dtype=dtype, device=device).view(3, 1, 1)
        denorm = [torch.clamp(img * std + mean, 0.0, 1.0) for img in augmented_images]

        grid = vutils.make_grid(torch.stack(denorm), nrow=min(4, len(denorm)))
        preview_path = self.cfg.paths.run_root / 'augmentation_preview.png'
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        vutils.save_image(grid, preview_path)
        self.logger.info(f'Saved augmentation preview with {len(denorm)} samples to {preview_path}')

    def _model_for_export(self) -> nn.Module:
        """Prefer the EMA weights when exporting/saving."""
        return self.ema_model.module if self.ema_model is not None else self.model

    def load_ema_state(self, ema_path: Path) -> None:
        """Restore EMA weights from disk if present."""
        if self.ema_model is None or not ema_path.exists():
            return
        state_dict = torch.load(ema_path, map_location=self.device)
        self.ema_model.load_state_dict(state_dict)
        self.ema_model.to(self.device)

    def _load_data(self, cfg: Config):
        """Materialise train/val datasets and wrap them in DataLoaders."""
        self.manager = SIDDatasetManager(
            dataset_name=cfg.data.dataset_name,
            use_streaming=cfg.data.use_streaming,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )

        train_transform = build_classification_transform(
            cfg.data.image_size,
            cfg.model.normalize_mean,
            cfg.model.normalize_std,
            cfg.data.augment,
            is_train=True,
        )
        val_transform = build_classification_transform(
            cfg.data.image_size,
            cfg.model.normalize_mean,
            cfg.model.normalize_std,
            cfg.data.augment,
            is_train=False,
        )

        train_ds = self.manager.get_split(
            split_type=TRAIN,
            max_samples=cfg.data.train_samples,
        )

        self._save_augmentation_preview(train_ds, train_transform)

        val_ds = self.manager.get_split(
            split_type=VALIDATION,
            max_samples=cfg.data.val_samples,
        )
        

        num_workers = cfg.loader.num_workers if not cfg.data.use_streaming else 0
        loader_common_kwargs = {
            "batch_size": cfg.loader.batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_common_kwargs["prefetch_factor"] = cfg.loader.prefetch_factor
            loader_common_kwargs["persistent_workers"] = cfg.loader.persistent_workers

        train_loader = DataLoader(
            SIDClassificationDataset(
                train_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                transform=train_transform,
                max_samples=cfg.data.train_samples,
                return_label=True,
            ),
            shuffle=cfg.loader.shuffle_train if not cfg.data.use_streaming else False, # In short, “use your configured shuffle setting when not streaming; disable DataLoader-level shuffling when streaming.”
            **loader_common_kwargs,
        )

        val_loader = DataLoader(
            SIDClassificationDataset(
                val_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                transform=val_transform,
                max_samples=cfg.data.val_samples,
                return_label=True,
            ),
            shuffle=False,
            **loader_common_kwargs,
        )
    
        return train_loader, val_loader
