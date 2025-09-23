import json, os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config

from .dataset_manager import SIDDatasetManager
from .dataset import SIDDataset
from .model import SimpleCNN
<<<<<<< HEAD
=======
from .utils import print_gpu_info, set_seed
>>>>>>> main
from .visualize import plot_training_curves
from .utils.logger import SidLogger as SidLogger
from .utils.model_manager import save_training_history

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
<<<<<<< HEAD

def train(logger: SidLogger, cfg: Config):
    device = torch.device(cfg.training.device)

    logger.log_training_config(cfg)

    manager = SIDDatasetManager(dataset_name=cfg.data.dataset_name, 
                                use_disk_cache=cfg.data.use_disk_cache, 
                                use_streaming=cfg.data.use_streaming)                             

=======

def train(cfg: Config):
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

    print("="*60)
    print("TRAINING MODE")
    print("="*60)
    print_gpu_info(device)
    print(f"Train samples: {cfg.data.train_samples}, Val samples: {cfg.data.val_samples}")
    print(f"Batch size: {cfg.loader.batch_size}, Epochs: {cfg.training.epochs}")
    print("="*60)

    manager = SIDDatasetManager(dataset_name=cfg.data.dataset_name, 
                                use_disk_cache=cfg.data.use_disk_cache, 
                                use_streaming=cfg.data.use_streaming)                             

>>>>>>> main
    train_ds, val_ds, _ = manager.get_splits(
        train_max=cfg.data.train_samples, 
        val_max=cfg.data.val_samples, test_max=0)
    
    train_loader = DataLoader(
        SIDDataset(
            train_ds,
            image_size=cfg.model.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std,
            device=device
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_train,
        num_workers=cfg.loader.num_workers
    )
    
    val_loader = DataLoader(
        SIDDataset(
            val_ds,
            image_size=cfg.model.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std,
            device=device
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_val,
        num_workers=cfg.loader.num_workers
    )

    model = SimpleCNN(num_classes=cfg.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

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
            learning_rate=cfg.training.learning_rate
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(cfg.paths.model_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.paths.model_path)
<<<<<<< HEAD
            logger.info(f"Saved best model (val_acc: {val_acc:.1f}%)")
=======
            print(f"Saved best model (val_acc: {val_acc:.1f}%)")
>>>>>>> main

    logger.log_training_complete(total_time=None, best_metric=best_val_acc, best_epoch=None)

<<<<<<< HEAD
    save_training_history(cfg.paths.results_dir, history, history_file='training_history.json')

    plot_training_curves(history)
    
    
=======
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    with open(os.path.join(cfg.paths.results_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history)
    print(f"Saved training history to {cfg.paths.results_dir}/training_history.json")
>>>>>>> main
