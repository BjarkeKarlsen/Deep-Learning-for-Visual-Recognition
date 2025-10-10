import math
import os
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake.data.dataset_manager import SIDDatasetManager
from deepfake.utils.model_persister import TorchModelPersister
from deepfake.segmentation.dataset import TamperedSegmentationDataset
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.visualization.segmentation_plots import SegmentationPlots
from deepfake.utils.logger import SidLogger
from deepfake.utils.training_metrics_tracker import TrainingMetrics, TrainingMetricsTracker
from deepfake.config import Config

def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Measure overlap between predicted and ground-truth masks (Dice score)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()  # 0.5 maps logits to the binary tamper mask expected downstream
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def prepare_dataloader(dataset, cfg: Config, *, shuffle: bool) -> DataLoader:
    """Wrap filtered HF splits with PyTorch loaders (or return None if empty)."""
    if dataset is None or len(dataset) == 0:
        return None

    wrapped = TamperedSegmentationDataset(
        dataset,
        image_size=cfg.data.image_size,
        normalize_mean=cfg.model.normalize_mean,
        normalize_std=cfg.model.normalize_std,
    )
    return DataLoader(
        wrapped,
        batch_size=cfg.loader.batch_size,
        shuffle=shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(logger: SidLogger, cfg: Config) -> Dict[str, float]:
    """Optimise the U-Net on tampered examples and log training history."""
    device = torch.device(cfg.training.device)
    logger.log_training_config(asdict(cfg))

    with SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_disk_cache=cfg.data.use_disk_cache,
        use_streaming=cfg.data.use_streaming,
    ) as manager:
        train_ds, val_ds, _ = manager.get_segmentation_splits(
            tampered_label=getattr(cfg.model, "tampered_label", 2),
            train_max=cfg.data.train_samples,
            val_max=cfg.data.val_samples,
            test_max=None,
        )

    train_loader = prepare_dataloader(train_ds, cfg, shuffle=cfg.loader.shuffle_train)
    # TODO: FIX THIS AND CREATE A NEW AND DO NOT THROW EXCEPTION. PREPARE DATALOADER SHOULD INSTEAD ALWAYS CREATE LEGIT DATA
    if train_loader is None:
        raise RuntimeError("No tampered samples with masks found for training")

    val_loader = prepare_dataloader(val_ds, cfg, shuffle=False)

    model = TamperSegmentationModel(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    metrics_tracker = TrainingMetricsTracker()

    metrics_tracker.start_training()
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{cfg.training.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(logits.detach(), masks).item()
            steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_train_loss = train_loss / max(steps, 1)
        avg_train_dice = train_dice / max(steps, 1)

        logger.info(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, train_dice={avg_train_dice:.4f}"
        )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_steps = 0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    logits = model(images)
                    loss = criterion(logits, masks)

                    val_loss += loss.item()
                    val_dice += dice_coefficient(logits, masks).item()
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            avg_val_dice = val_dice / max(val_steps, 1)
        else:
            avg_val_loss = float("nan")
            avg_val_dice = float("nan")

        metrics_tracker.add_metrics(TrainingMetrics(
            epoch=epoch,
            step=steps,  # or global_step
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            learning_rate=optimizer.param_groups[0]['lr'],
            additional_metrics={
                "train_dice": avg_train_dice,
                "val_dice": avg_val_dice,
                #"train_iou": avg_train_iou,
                #"val_iou": avg_val_iou,
            }
        ))

 
        logger.info(
            f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f}, val_dice={avg_val_dice:.4f}"
        )

        dice_to_use = avg_val_dice if not math.isnan(avg_val_dice) else avg_train_dice

        best_entry = metrics_tracker.get_best_metric("val_dice")
        best_val = (best_entry.additional_metrics["val_dice"]
                    if best_entry and not math.isnan(best_entry.additional_metrics["val_dice"])
                    else float('-inf'))

        if dice_to_use > best_val:
            TorchModelPersister().save_model(model, cfg.paths.model_path)
            logger.info(f"Saved best segmentation model (dice={dice_to_use:.4f})")
                        

    metrics_tracker.end_training()
    metrics_tracker.save_to_json(cfg.paths.history_path)

    summary = metrics_tracker.get_summary_stats()
    logger.log_list_of_dicts(f"Training Summary", summary)

    plotter = SegmentationPlots(output_directory=cfg.paths.run_root, training_history_path=cfg.paths.history_path)

    plotter.plot_training_history()
    plotter.plot_learning_rate_schedule()
