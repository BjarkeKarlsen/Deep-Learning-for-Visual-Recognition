import json, os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import SID_Logger
from .dataset_manager import SIDDatasetManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, LOGGING_LEVEL,
    NUM_CLASSES, NUM_WORKERS, RESULTS_DIR, SEED, TRAIN_SAMPLES, VAL_SAMPLES
)
from .dataset import SIDDataset
from .model import MultiTaskCNN
from .utils.utils import get_device, print_usage, set_seed, iou_metric
from .visualize import plot_training_curves

def train():
    set_seed(SEED)
    device = get_device()
    
    sid_logger = SID_Logger(
    name="training",
    log_dir="training_logs",
    log_level=LOGGING_LEVEL,
    console_output=True,
    file_output=True,
    )

    sid_logger.info(f"Using device: {device}")

    manager = SIDDatasetManager(use_disk_cache=True, use_streaming=True)
    
    train_ds, val_ds, _ = manager.get_splits(
        train_max=TRAIN_SAMPLES, val_max=VAL_SAMPLES, test_max=0
    )
    train_loader = DataLoader(
        SIDDataset(train_ds, device), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        SIDDataset(val_ds, device), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    model = MultiTaskCNN(num_classes=NUM_CLASSES).to(device)
    cls_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sid_logger.log_model_info(model, total_params, trainable_params)

    config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss_functions": "CrossEntropy + BCEWithLogits"
    }
    sid_logger.log_training_config(config)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_iou': []}

    for epoch in range(EPOCHS):
        sid_logger.log_epoch_start(epoch, EPOCHS)

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            images = batch["image"]  # shape: (B,3,256,256)
            masks  = batch["mask"]   # shape: (B,1,256,256)
            labels = batch["label"]  # shape: (B,)
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()
            class_logits, mask_logits = model(images)

            # classification loss
            loss_cls = cls_loss_fn(class_logits, labels)
            # segmentation loss
            loss_seg = seg_loss_fn(mask_logits, masks)
            # combined loss
            loss = loss_cls + loss_seg
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(class_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            #sid_logger.log_training_progress(epoch, EPOCHS, train_loss, train_correct, train_total)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        total_iou = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                labels = batch["label"].to(device)
                
                class_logits, mask_logits = model(images)
                loss = cls_loss_fn(class_logits, labels) + seg_loss_fn(mask_logits, masks)

                # IoU metric for segmentation just in case there’s any residual non-binary noise (e.g., interpolation artifacts), force them to exactly 0 or 1:
                masks_bin = (masks > 0.5).float()
                preds = (torch.sigmoid(mask_logits) > 0.5).float()
                total_iou += iou_metric(preds, masks_bin)

                val_loss += loss.item()
                _, predicted = torch.max(class_logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(avg_iou)
        
        sid_logger.log_epoch_results(
            epoch, avg_train_loss, train_acc,
            avg_val_loss, val_acc, LEARNING_RATE
        )

        #print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
        #      f"Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%, Val IoU: {avg_iou:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (val_acc: {val_acc:.1f}%)")

    sid_logger.log_training_complete(total_time=0.0, best_metric=best_val_acc, best_epoch=None)
    print(f"\nTraining complete! Best validation: {best_val_acc:.1f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history)
    sid_logger.info(f"Saved training history to {RESULTS_DIR}/training_history.json")
    #print(f"Saved training history to {RESULTS_DIR}/training_history.json")
    
