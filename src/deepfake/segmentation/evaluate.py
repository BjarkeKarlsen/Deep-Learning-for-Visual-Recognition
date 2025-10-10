import math
from dataclasses import asdict
import os
from typing import Dict

from deepfake.visualization.segmentation_plots import SegmentationPlots
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake.data.dataset_manager import SIDDatasetManager
from deepfake.segmentation.dataset import TamperedSegmentationDataset
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.utils.evaluation_metrics_tracker import SegmentationEvaluationMetrics, EvaluationMetricsTracker
from deepfake.utils.logger import SidLogger
from deepfake.utils.model_manager import check_model_exists
from deepfake.utils.model_persister import TorchModelPersister
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

@torch.no_grad()
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

    metrics_tracker = EvaluationMetricsTracker(SegmentationEvaluationMetrics, logger=logger)


    model = TamperSegmentationModel(in_channels=3, out_channels=1).to(device)
    model_persister = TorchModelPersister()
    model_persister.load_model(model, cfg.paths.model_path)
    model.eval()
    
    gallery = []

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
            max_examples = 6
            if len(gallery) < max_examples:
                # Get raw image (denormalize)
                raw_img = images[0].cpu()
                mean = torch.tensor(cfg.model.normalize_mean).view(3, 1, 1)
                std = torch.tensor(cfg.model.normalize_std).view(3, 1, 1)
                denorm_img = (raw_img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                
                # Get masks (squeeze channel dimension)
                true_mask = masks[0].squeeze(0).cpu().numpy()  # Remove channel dim
                pred_mask = (torch.sigmoid(logits[0]) > 0.5).squeeze(0).cpu().numpy().astype(float)
                
                gallery.append((denorm_img, true_mask, pred_mask))

    mean_dice = float(np.mean(dice_scores)) if dice_scores else float("nan")
    mean_iou = float(np.mean(iou_scores)) if iou_scores else float("nan")

     # Instead of passing a dict, create a proper metrics object:
    metrics_tracker.add_metrics(SegmentationEvaluationMetrics(
        task_type="segmentation",
        primary_metric="dice",
        primary_score=mean_dice,
        dice_coefficient=mean_dice,
        mean_iou=mean_iou
    ))
    
    logger.info(f"Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}")

    results = {"dice_coefficient": mean_dice, "mean_iou": mean_iou}
    logger.save_json(results, "segmentation_evaluation.json")
    evaluation_metrics_path = os.path.join(cfg.paths.run_root, "evaluation_metrics.json")
    # plotter = SegmentationPlots()
    # for idx, (im, true_m, pred_m) in enumerate(gallery, start=1):
    #     save_path = cfg.paths.output_dir / f"segmentation_example_{idx}.png"
    #     plotter.plot_segmentation(im, true_m, pred_m, save_path=str(save_path))
    plotter = SegmentationPlots(output_directory=str(cfg.paths.run_root), eval_history_path=str(evaluation_metrics_path))
    gallery_path = cfg.paths.run_root / "segmentation_gallery.png"
    plotter.plot_segmentation_gallery(gallery, ncols=3, save_path=str(gallery_path))
    metrics_tracker.save_to_json(evaluation_metrics_path)

    return results
