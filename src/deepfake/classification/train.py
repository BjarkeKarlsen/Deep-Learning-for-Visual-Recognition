from dataclasses import asdict
from deepfake.utils.model_manager import check_model_exists
from deepfake.utils.model_persister import TorchModelPersister
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode

from deepfake.config import Config
from deepfake.data.dataset_manager import SIDDatasetManager, TRAIN, VALIDATION
from deepfake.utils.logger import SidLogger as SidLogger
from deepfake.utils.model_persister import TorchModelPersister
from deepfake.utils.optimizer_factory import build_optimizer
from deepfake.utils.training_metrics_tracker import TrainingMetrics, TrainingMetricsTracker
from deepfake.visualization.classification_plots import ClassificationPlots
from .dataset import SIDClassificationDataset
from .model import BaselineClassifier



def train(logger: SidLogger, cfg: Config):
    """Train the lightweight classifier on the configured SID subsets."""
    device = torch.device(cfg.training.device)

    logger.log_training_config(asdict(cfg))

    metrics_tracker = TrainingMetricsTracker()

    if os.path.isfile(cfg.paths.history_path):
        metrics_tracker.load_from_json(cfg.paths.history_path)

    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_streaming=cfg.data.use_streaming,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )

    train_ds = manager.get_split(
        split_type=TRAIN,
        max_samples=cfg.data.train_samples,
    )
    
    val_ds = manager.get_split(
        split_type=VALIDATION,
        max_samples=cfg.data.val_samples,
    )
    
    # For streaming, shuffle via the HF API, not DataLoader:
    if cfg.data.use_streaming:
        # buffer_size can be your train_samples or a smaller window
        train_ds = train_ds.shuffle(buffer_size=cfg.data.train_samples, seed=cfg.training.seed)

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
        shuffle=cfg.loader.shuffle_val,
        num_workers=cfg.loader.num_workers,
    )


    model = BaselineClassifier(num_classes=cfg.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    if check_model_exists(cfg.paths.model_path):
        loader = TorchModelPersister()
        loader.load_model(model, cfg.paths.model_path)
    optimizer = build_optimizer(model.parameters(), cfg.training)

    best_val_acc = None
    best_epoch = None
    metrics_tracker.start_training()
    
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

        train_acc = train_correct / train_total if train_total else float('nan')
        val_acc = val_correct / val_total if val_total else float('nan')
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = (
            val_loss / len(val_loader)
            if len(val_loader) > 0
            else float('nan')
        )
        
   
        metrics_tracker.add_metrics(TrainingMetrics(
            epoch=epoch,
            step=(epoch + 1) * len(train_loader),
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            learning_rate=optimizer.param_groups[0]['lr'],
        ))

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
                model_persister = TorchModelPersister()
                model_persister.save_model(model, cfg.paths.model_path)
                logger.info(f"Saved best model (val_acc: {val_acc:.1f}%)")


    metrics_tracker.end_training()
    
    logger.log_training_complete(
        total_time=metrics_tracker.get_summary_stats().get('duration_minutes'),
        best_metric=best_val_acc,
        best_epoch=best_epoch,
    )
    
    metrics_tracker.save_to_json(cfg.paths.history_path)
    
    summary = metrics_tracker.get_summary_stats()
    logger.info(f"Training Summary: {summary}")

    
    plotter = ClassificationPlots(output_directory=cfg.paths.run_root, training_history_path=cfg.paths.history_path)

    plotter.plot_training_history()
    plotter.plot_learning_rate_schedule()
