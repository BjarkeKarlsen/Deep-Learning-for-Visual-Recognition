import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Config
from src.common.dataset_manager import SIDDatasetManager
from src.common.visualize import plot_training_curves
from src.utils.logger import SidLogger as SidLogger
from src.utils.model_manager import save_training_history

from .dataset import SIDDataset
from .model import SimpleCNN

def train(logger: SidLogger, cfg: Config):
    device = torch.device(cfg.training.device)

    logger.log_training_config(cfg)

    manager = SIDDatasetManager(dataset_name=cfg.data.dataset_name, 
                                use_disk_cache=cfg.data.use_disk_cache, 
                                use_streaming=cfg.data.use_streaming)                             

    train_ds, val_ds, _ = manager.get_splits(
        train_max=cfg.data.train_samples, 
        val_max=cfg.data.val_samples, test_max=0)
    
    train_loader = DataLoader(
        SIDDataset(
            train_ds,
            image_size=cfg.data.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_train,
        num_workers=cfg.loader.num_workers
    )
    
    val_loader = DataLoader(
        SIDDataset(
            val_ds,
            image_size=cfg.data.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_val,
        num_workers=cfg.loader.num_workers
    )

    model = SimpleCNN(num_classes=cfg.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    best_val_acc = None
    best_epoch = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total if train_total else float('nan')
        val_acc = 100 * val_correct / val_total if val_total else float('nan')
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = (
            val_loss / len(val_loader)
            if len(val_loader) > 0
            else float('nan')
        )

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        logger.log_epoch_results(
            epoch, 
            train_loss=avg_train_loss, 
            train_acc=train_acc, 
            val_loss=avg_val_loss, 
            val_acc=val_acc, 
        )

        if val_total > 0:
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                os.makedirs(os.path.dirname(cfg.paths.model_path), exist_ok=True)
                torch.save(model.state_dict(), cfg.paths.model_path)
                logger.info(f"Saved best model (val_acc: {val_acc:.1f}%)")

    logger.log_training_complete(
        total_time=None,
        best_metric=best_val_acc,
        best_epoch=best_epoch,
    )

    save_training_history(cfg.paths.results_dir, history, history_file='training_history.json')

    plot_training_curves(history, output_dir=cfg.paths.results_dir)

    
