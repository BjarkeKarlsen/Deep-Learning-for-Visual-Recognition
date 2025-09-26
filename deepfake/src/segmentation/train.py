import math
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.dataset_manager import SIDDatasetManager
from src.common.visualize import plot_segmentation_curves
from src.segmentation.dataset import TamperedSegmentationDataset
from src.segmentation.model import SimpleUNet
from src.utils.logger import SidLogger
from src.utils.model_manager import save_training_history
from src.config import Config


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def prepare_dataloader(dataset, cfg: Config, *, shuffle: bool) -> DataLoader:
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
    device = torch.device(cfg.training.device)
    logger.log_training_config(cfg)

    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_disk_cache=cfg.data.use_disk_cache,
        use_streaming=cfg.data.use_streaming,
    )

    train_ds, val_ds, _ = manager.get_segmentation_splits(
        tampered_label=getattr(cfg.model, "tampered_label", 2),
        train_max=cfg.data.train_samples,
        val_max=cfg.data.val_samples,
        test_max=None,
    )

    train_loader = prepare_dataloader(train_ds, cfg, shuffle=cfg.loader.shuffle_train)
    if train_loader is None:
        raise RuntimeError("No tampered samples with masks found for training")

    val_loader = prepare_dataloader(val_ds, cfg, shuffle=False)

    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    best_val_dice = None
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

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

        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(avg_train_dice)

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

        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)

        logger.info(
            f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f}, val_dice={avg_val_dice:.4f}"
        )

        metric_to_compare = avg_val_dice if not math.isnan(avg_val_dice) else avg_train_dice
        if best_val_dice is None or metric_to_compare > best_val_dice:
            best_val_dice = metric_to_compare
            model_dir = os.path.dirname(cfg.paths.model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), cfg.paths.model_path)
            logger.info(f"Saved best segmentation model (dice={metric_to_compare:.4f})")

    save_training_history(
        cfg.paths.results_dir,
        history,
        history_file=cfg.paths.history_file,
    )
    plot_segmentation_curves(history, output_dir=cfg.paths.results_dir)
    return {
        "best_dice": best_val_dice if best_val_dice is not None else float("nan"),
        "epochs": cfg.training.epochs,
    }
