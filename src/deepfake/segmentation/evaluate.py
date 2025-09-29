import math
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake.data.dataset_manager import SIDDatasetManager
from deepfake.segmentation.dataset import TamperedSegmentationDataset
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.utils.logger import SidLogger
from deepfake.utils.model_manager import check_model_exists, save_training_history
from deepfake.config import Config


def dice_and_iou(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> Dict[str, torch.Tensor]:
    """Compute dataset-level overlap metrics for binary tamper masks."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()  # 0.5 aligns with the binary tamper-versus-background assumption
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    overlap = (preds * targets).sum(dim=(1, 2, 3))
    union_iou = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - overlap
    iou = (overlap + eps) / (union_iou + eps)
    return {"dice": dice.mean(), "iou": iou.mean()}


def prepare_dataloader(dataset, cfg: Config) -> DataLoader:
    """Wrap a segmentation subset with transforms or return `None` when empty."""
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
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(logger: SidLogger, cfg: Config) -> Dict[str, float]:
    """Evaluate a trained U-Net and persist Dice/IoU summaries."""
    device = torch.device(cfg.training.device)
    logger.log_evaluation_config(asdict(cfg))

    check_model_exists(cfg.paths.model_path)

    with SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_disk_cache=cfg.data.use_disk_cache,
        use_streaming=cfg.data.use_streaming,
    ) as manager:
        _, _, test_ds = manager.get_segmentation_splits(
            tampered_label=getattr(cfg.model, "tampered_label", 2),
            train_max=0,
            val_max=0,
            test_max=cfg.data.test_samples,
            test_offset=cfg.data.val_samples,
        )

    test_loader = prepare_dataloader(test_ds, cfg)
    if test_loader is None:
        raise RuntimeError("No tampered samples with masks available for evaluation")

    model = TamperSegmentationModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(cfg.paths.model_path, map_location=device))
    model.eval()

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Seg Eval"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metrics = dice_and_iou(logits, masks)
            dice_scores.append(metrics["dice"].item())
            iou_scores.append(metrics["iou"].item())

    mean_dice = float(np.mean(dice_scores)) if dice_scores else float("nan")
    mean_iou = float(np.mean(iou_scores)) if iou_scores else float("nan")

    logger.info(f"Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}")

    results = {"dice": mean_dice, "iou": mean_iou}
    save_training_history(cfg.paths.results_dir, results, history_file="segmentation_eval.json")
    logger.save_json(results, "segmentation_evaluation.json")

    return results
