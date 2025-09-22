import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils.logger import SID_Logger
from .dataset_manager import SIDDatasetManager
from .dataset import SIDDataset
from .model import MultiTaskCNN
from .utils.utils import set_seed, get_device, iou_metric
from .visualize import plot_training_curves
from .visualization.visualize_enhanced import plot_enhanced_training_curves
from .loss_factory import LossFactory
from .metric.metric import iou_metric, dice_metric
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, LOGGING_LEVEL, LOSS_CONFIG,
    NUM_CLASSES, NUM_WORKERS, RESULTS_DIR, SEED, TRAIN_SAMPLES, VAL_SAMPLES
)

class Trainer:
    def __init__(self):
        # 1. Config & logger
        set_seed(SEED)
        self.device = get_device()
        self.logger = SID_Logger(
            name="training",
            log_dir="training_logs",
            log_level=LOGGING_LEVEL,
        )
        self.logger.info(f"Using device: {self.device}")

        # 2. Data loaders
        manager = SIDDatasetManager(use_disk_cache=True, use_streaming=True)
        train_ds, val_ds, _ = manager.get_splits(
            train_max=TRAIN_SAMPLES, val_max=VAL_SAMPLES, test_max=0
        )
        self.train_loader = DataLoader(
            SIDDataset(train_ds, self.device),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        self.val_loader = DataLoader(
            SIDDataset(val_ds, self.device),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        # 3. Model, loss, optimizer
        self.model = MultiTaskCNN(num_classes=NUM_CLASSES).to(self.device)
        self.cls_loss = nn.CrossEntropyLoss()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # 4. Log model & config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_model_info(self.model, total_params, trainable_params)

        self.loss_config = LOSS_CONFIG or {
            'cls_loss_type': 'crossentropy',
            'seg_loss_type': 'bce',
            'cls_weight': 1.0,
            'seg_weight': 1.0
        }
        
        self.loss_fn = LossFactory.create_combined_loss(
            cls_loss_type=self.loss_config['cls_loss_type'],
            seg_loss_type=self.loss_config['seg_loss_type'],
            cls_weight=self.loss_config['cls_weight'],
            seg_weight=self.loss_config['seg_weight']
        )
        
        config = {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "classification_loss": self.loss_config['cls_loss_type'],
            "segmentation_loss": self.loss_config['seg_loss_type'],
            "cls_weight": self.loss_config['cls_weight'],
            "seg_weight": self.loss_config['seg_weight']
        }
        self.logger.log_training_config(config)

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        correct, total = 0, 0

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}/{EPOCHS}")
        ):
            imgs = batch["image"]#.to(self.device) # Implicit done
            masks = batch["mask"]#.to(self.device)
            labels = batch["label"]#.to(self.device)

            self.optimizer.zero_grad()
            logits_cls, logits_seg = self.model(imgs)
            
            # Use combined loss function
            total_loss, cls_loss, seg_loss = self.loss_fn(logits_cls, logits_seg, labels, masks)
            
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_seg_loss += seg_loss.item()
            
            preds = torch.argmax(logits_cls, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            self.logger.log_batch_progress(
                batch_idx, len(self.train_loader), total_loss.item(), BATCH_SIZE, log_interval=50
            )

        avg_loss = running_loss / len(self.train_loader)
        avg_cls_loss = running_cls_loss / len(self.train_loader)
        avg_seg_loss = running_seg_loss / len(self.train_loader)
        acc = 100 * correct / total
        return avg_loss, avg_cls_loss, avg_seg_loss, acc

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_seg_loss = 0.0
        correct = 0.0
        total = 0.0
        total_iou = 0.0
        total_dice = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch["image"]#.to(self.device)
                masks = batch["mask"]#.to(self.device)
                labels = batch["label"]#.to(self.device)

                logits_cls, logits_seg = self.model(imgs)
                
                total_loss, cls_loss, seg_loss = self.loss_fn(logits_cls, logits_seg, labels, masks)
                
                val_loss += total_loss.item()
                val_cls_loss += cls_loss.item()
                val_seg_loss += seg_loss.item()

                preds = torch.argmax(logits_cls, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Compute segmentation metrics
                masks_bin = (masks > 0.5).float()
                preds_bin = (torch.sigmoid(logits_seg) > 0.5).float()
                total_iou += iou_metric(preds_bin, masks_bin)
                total_dice += dice_metric(preds_bin, masks_bin)

        avg_loss = val_loss / len(self.val_loader)
        avg_cls_loss = val_cls_loss / len(self.val_loader)
        avg_seg_loss = val_seg_loss / len(self.val_loader)
        acc = 100 * correct / total
        avg_iou = total_iou / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        return avg_loss, avg_cls_loss, avg_seg_loss, acc, avg_iou, avg_dice

    def save_history_and_plots(self, history):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history)
        plot_enhanced_training_curves(history)
        self.logger.info(f"Saved training history to {RESULTS_DIR}/training_history.json")

    def run(self):
        best_val_acc = 0.0
        history = {
            "train_loss": [], "train_cls_loss": [], "train_seg_loss": [], "train_acc": [],
            "val_loss": [], "val_cls_loss": [], "val_seg_loss": [], "val_acc": [], 
            "val_iou": [], "val_dice": []
        }

        for epoch in range(EPOCHS):
            self.logger.log_epoch_start(epoch, EPOCHS)
            # Training
            t_loss, t_cls_loss, t_seg_loss, t_acc = self.train_epoch(epoch)
            
            # Validation
            v_loss, v_cls_loss, v_seg_loss, v_acc, v_iou, v_dice = self.validate_epoch()

            history["train_loss"].append(t_loss)
            history["train_cls_loss"].append(t_cls_loss)
            history["train_seg_loss"].append(t_seg_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_cls_loss"].append(v_cls_loss)
            history["val_seg_loss"].append(v_seg_loss)
            history["val_acc"].append(v_acc)
            history["val_iou"].append(v_iou)
            history["val_dice"].append(v_dice)

            self.logger.log_epoch_results(epoch, t_loss, t_acc, v_loss, v_acc, LEARNING_RATE)
            
            self.logger.info(
                f"Detailed - Train: cls_loss={t_cls_loss:.4f}, seg_loss={t_seg_loss:.4f} | "
                f"Val: cls_loss={v_cls_loss:.4f}, seg_loss={v_seg_loss:.4f}, "
                f"IoU={v_iou:.4f}, Dice={v_dice:.4f}"
            )

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(self.model.state_dict(), MODEL_PATH)
                self.logger.log_best_model(epoch, "Validation Accuracy", v_acc)

        self.logger.log_training_complete(total_time=0.0, best_metric=best_val_acc, best_epoch=None)
        self.save_history_and_plots(history)


if __name__ == "__main__":
    Trainer().run()
