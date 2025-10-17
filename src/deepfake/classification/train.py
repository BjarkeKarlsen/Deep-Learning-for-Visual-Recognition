import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode
from dataclasses import asdict
from torch.amp import autocast, GradScaler

from deepfake.config import Config
from deepfake.utils.optimizer_factory import OptimizerFactory
from deepfake.utils.model_persister import TorchModelPersister
from deepfake.data.dataset_manager import SIDDatasetManager, TRAIN, VALIDATION
from deepfake.utils.logger import SidLogger as SidLogger
from deepfake.utils.model_persister import TorchModelPersister, IModelPersister
from deepfake.utils.training_metrics_tracker import TrainingMetrics, TrainingMetricsTracker
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
        self.logger = logger
        self.metrics_tracker = metrics_tracker
        self.checkpoint_mgr = checkpoint_mgr
        self.persister = persister or TorchModelPersister()
        
        
        # Model, optimizer, criterion, scaler
        self.model = BaselineClassifier(num_classes=cfg.model.num_classes).to(self.device)
        self.optimizer = OptimizerFactory(self.model.parameters(), cfg.training)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

        
         # Load best model if exists
        if os.path.isfile(cfg.paths.model_path):
            self.persister.load_model(self.model, cfg.paths.model_path)
            self.logger.info(f"Loaded best model from {cfg.paths.model_path}")       
        
        self.logger.log_training_config(asdict(cfg))

    def train(self, start_epoch: int = 0):
        """Train the lightweight classifier on the configured SID subsets."""
        prev_best = None
        prev_best_val = float('-inf')
        
        # LOAD THE DATA
        train_loader, val_loader = self._load_data(self.cfg)

        # START TRAINING
        self.metrics_tracker.start_training()

        for epoch in range(start_epoch, self.cfg.training.epochs):
            # TRAIN & EVAL
            train_acc, avg_train_loss = self._train_epoch(train_loader, epoch)
            val_acc, avg_val_loss = self._evaluation(val_loader)

            # DETERMINE PREVIOUS BEST BEFORE RECORDING CURRENT METRICS
            prev_best = self.metrics_tracker.get_best_metric("val_acc")
            prev_best_val = prev_best.val_acc if prev_best and prev_best.val_acc is not None else float('-inf')
            
            # RECORD METRICS
            self.metrics_tracker.add_metrics(TrainingMetrics(
                epoch=self.metrics_tracker.metrics[-1].epoch + 1,
                step=(self.metrics_tracker.metrics[-1].epoch + 1) * len(train_loader),
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=self.optimizer.param_groups[0]['lr'],
            ))
            
            # LOGGING EPOCH
            self.logger.log_epoch_results(
                epoch,
                train_loss=avg_train_loss,
                train_acc=train_acc,
                val_loss=avg_val_loss,
                val_acc=val_acc,
            )
                
            # SAVE BEST MODEL IF IMPROVED
            if val_acc > prev_best_val:
                self.persister.save_model(self.model, self.cfg.paths.model_path)
                self.logger.info(f"Saved best model at epoch {epoch+1} (val_acc={val_acc:.4f})")
                
            # CHECKPOINT        
            self.checkpoint_mgr.create_checkpoint(
                epoch=self.metrics_tracker.metrics[-1].epoch + 1,
                model=self.model,
                optimizer=self.optimizer,
                metrics_tracker=self.metrics_tracker,
                config=self.cfg,
            )

        # FINISH
        self.metrics_tracker.end_training()
        self.metrics_tracker.save_to_json(self.cfg.paths.history_path)

        self.logger.log_training_complete(
            total_time=self.metrics_tracker.get_summary_stats().get('duration_minutes'),
            best_metric=self.metrics_tracker.get_best_metric("val_acc").val_acc,
            best_epoch=self.metrics_tracker.get_best_metric("val_acc").epoch,
        )

        # PLOT
        plotter = ClassificationPlots(output_directory=self.cfg.paths.run_root, training_history_path=self.cfg.paths.history_path)
        plotter.plot_training_history()
        plotter.plot_learning_rate_schedule()
        

    def _train_epoch(self, train_loader: DataLoader, epoch: int = 0):
        self.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # ITERATE OVER BATCHES WITH A PROGRESS BAR
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.training.epochs}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # ZERO GRADIENTS BEFORE BACKWARD PASS
            self.optimizer.zero_grad()
            
            # ENABLE MIXED PRECISION FOR FASTER FORWARD AND REDUCED MEMORY
            with autocast(device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # SCALE LOSS AND BACKWARD FOR STABLE MIXED-PRECISION TRAINING    
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # UPDATE TRAINING METRICS
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # UPDATE PROGRESS BAR WITH CURRENT LOSS
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        # CALCULATE AVERAGE METRICS    
        train_acc = train_correct / train_total if train_total else float('nan')
        avg_train_loss = train_loss / len(train_loader)
        
        return train_acc, avg_train_loss
    
    @torch.no_grad() # Optimize memory usage and speed up computations
    def _evaluation(self, val_loader: DataLoader):

        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # FORWARD PASS THROUGH CLASSIFICATION MODEL
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # ACCUMULATE LOSS
                val_loss += loss.item()
                
                # COUNT TOTAL SAMPLES AND CORRECT PREDICTIONS
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # CALCULATE ACCURACY AND AVERAGE LOSS WITH SAFE DIVISION
        val_acc = val_correct / val_total if val_total > 0 else float('nan')
        avg_val_loss = (val_loss / len(val_loader) if len(val_loader) > 0 else float('nan'))

        return  val_acc, avg_val_loss
    
    
        
    def _load_data(self, cfg: Config):
        self.manager = SIDDatasetManager(
            dataset_name=cfg.data.dataset_name,
            use_streaming=cfg.data.use_streaming,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )

        train_ds = self.manager.get_split(
            split_type=TRAIN,
            max_samples=cfg.data.train_samples,
        )
        self.manager = SIDDatasetManager(
            dataset_name=cfg.data.dataset_name,
            use_streaming=cfg.data.use_streaming,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )

        train_ds = self.manager.get_split(
            split_type=TRAIN,
            max_samples=cfg.data.train_samples,
        )
        
        val_ds = self.manager.get_split(
            split_type=VALIDATION,
            max_samples=cfg.data.val_samples,
        )
        

        train_loader = DataLoader(
            SIDClassificationDataset(
                train_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                max_samples=cfg.data.train_samples,
                return_label=True,
            ),
            batch_size=cfg.loader.batch_size,
            shuffle=cfg.loader.shuffle_train if not cfg.data.use_streaming else False, # In short, “use your configured shuffle setting when not streaming; disable DataLoader-level shuffling when streaming.”
            num_workers=cfg.loader.num_workers if not cfg.data.use_streaming else 0, # Multiprocessing with streaming datasets is not supported
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            SIDClassificationDataset(
                val_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                max_samples=cfg.data.val_samples,
                return_label=True,
            ),
            batch_size=cfg.loader.batch_size,
            shuffle=False,
            num_workers=cfg.loader.num_workers if not cfg.data.use_streaming else 0,
            pin_memory=torch.cuda.is_available(),
        )
    
        return train_loader, val_loader
