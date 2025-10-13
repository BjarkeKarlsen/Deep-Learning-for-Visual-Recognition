import math
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode
from torch.amp import autocast, GradScaler


from deepfake.config import Config
from deepfake.data.dataset_manager import DatasetFilters, SIDDatasetManager, TRAIN, VALIDATION
from deepfake.utils.model_persister import TorchModelPersister
from deepfake.data.dataset import SIDClassificationDataset
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.utils.logger import SidLogger
from deepfake.utils.model_persister import TorchModelPersister
from deepfake.utils.optimizer_factory import build_optimizer
from deepfake.utils.training_metrics_tracker import TrainingMetrics, TrainingMetricsTracker
from deepfake.visualization.segmentation_plots import SegmentationPlots

def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the Dice Coefficient over a batch:
      Inputs:
        logits:  Tensor of shape (N,1,H,W)
        targets: Tensor of shape (N,1,H,W)
    Returns:
        single scalar Dice score across all N×H×W pixels
    """
    # 1. Convert logits → probabilities
    probs = torch.sigmoid(logits)
    # 2. Binarize predictions and targets
    preds = (probs > 0.5).float()
    targs = (targets > 0.5).float()

    # 3. Flatten batch and spatial dims → (N*H*W)
    preds_flat = preds.view(-1)
    targs_flat = targs.view(-1)

    # 4. Compute intersection and union
    intersection = (preds_flat * targs_flat).sum()
    union = preds_flat.sum() + targs_flat.sum()

    # 5. Dice Coefficient
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def train(logger: SidLogger, cfg: Config) -> Dict[str, float]:
    """Optimise the U-Net on tampered examples and log training history."""
    device = torch.device(cfg.training.device)
    logger.log_training_config(asdict(cfg))

    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_streaming=cfg.data.use_streaming,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )

    train_ds = manager.get_split(
        split_type=TRAIN,
        max_samples=cfg.data.train_samples,
        filter_fn=DatasetFilters.tampered_with_masks
    )
    
    val_ds = manager.get_split(
        split_type=VALIDATION,
        max_samples=cfg.data.val_samples,
        filter_fn=DatasetFilters.tampered_with_masks
    )
    
    train_loader = DataLoader(
        SIDClassificationDataset(
            train_ds,
            image_size=cfg.data.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std,
            return_mask=True,
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        SIDClassificationDataset(
            val_ds,
            image_size=cfg.data.image_size,
            normalize_mean=cfg.model.normalize_mean,
            normalize_std=cfg.model.normalize_std,
            return_mask=True,
        ),
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


    model = TamperSegmentationModel(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model.parameters(), cfg.training)
    
    metrics_tracker = TrainingMetricsTracker()
    scaler = GradScaler(device=cfg.training.device)  # For scaling gradients

    metrics_tracker.start_training()
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{cfg.training.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=cfg.training.device):  # Enable mixed precision
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()

            # Gradient clipping is applied after backward() and before optimizer step when using mixed precision.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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
                for batch in val_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
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
            
        latest_best_metrics = metrics_tracker.get_best_metric("val_dice")
        prev_best_val = latest_best_metrics.additional_metrics.get("val_dice") if latest_best_metrics else float('-inf')

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
 
        if avg_val_dice > prev_best_val:
            TorchModelPersister().save_model(model, cfg.paths.model_path)
            logger.info(
                f"Saved best segmentation model at epoch {epoch} "
                f"(dice={avg_val_dice:.4f})"
            )
                            

    metrics_tracker.end_training()
    metrics_tracker.save_to_json(cfg.paths.history_path)

    summary = metrics_tracker.get_summary_stats()
    logger.log_list_of_dicts(f"Training Summary", summary)

    plotter = SegmentationPlots(output_directory=cfg.paths.run_root, training_history_path=cfg.paths.history_path)

    plotter.plot_training_history()
    plotter.plot_learning_rate_schedule()
