from dataclasses import asdict
from typing import Dict, List, Tuple

from deepfake.data.dataset import SIDClassificationDataset
from deepfake.visualization.segmentation_plots import SegmentationPlots
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DownloadMode

from deepfake.data.dataset_manager import TEST, DatasetFilters, SIDDatasetManager
from deepfake.segmentation.model import TamperSegmentationModel
from deepfake.utils.evaluation_metrics_tracker import SegmentationEvaluationMetrics, EvaluationMetricsTracker
from deepfake.utils.logger import SidLogger
from deepfake.utils.model_persister import TorchModelPersister, IModelPersister
from deepfake.config import Config
from deepfake.utils.augmentation_factory import build_segmentation_transforms

EPS = 1e-7


class Evaluator:
    """
    Runs segmentation inference, records Dice/IoU metrics, and generates gallery and plots.
    """
    # DRIVES SEGMENTATION EVALUATION, SUMMARY METRICS, AND VISUAL OUTPUTS.

    def __init__(
        self,
        cfg: Config,
        logger: SidLogger,
        persister: IModelPersister = None
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.training.device)
        eval_cfg = getattr(cfg, "evaluation", None)
        self.thresholds = sorted(set(getattr(eval_cfg, "thresholds", None) or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        self.max_examples = getattr(eval_cfg, "analysis_examples", 6) if eval_cfg else 6
        bucket_edges = getattr(eval_cfg, "bucket_edges", None) or [0.5, 2.0]
        self.bucket_bounds = [0.0] + sorted(bucket_edges) + [float("inf")]
        self.bucket_labels = self._build_bucket_labels(self.bucket_bounds)

        # DATASET & DATALOADER
        # BUILD A TEST SPLIT THAT INCLUDES TAMPERED MASKS AND APPLIES THE SAME TRANSFORMS USED DURING TRAINING.
        manager = SIDDatasetManager(
            dataset_name=cfg.data.dataset_name,
            use_streaming=cfg.data.use_streaming,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )
        test_ds = manager.get_split(
            split_type=TEST,
            max_samples=cfg.data.test_samples,
            use_test_or_val_as_test_set=True,
            val_offset=cfg.data.val_samples,
            filter_fn=DatasetFilters.tampered_with_masks,
        )
        joint_tf, image_tf, mask_tf = build_segmentation_transforms(
            cfg.data.image_size,
            cfg.model.normalize_mean,
            cfg.model.normalize_std,
            cfg.data.augment,
            is_train=False,
        )
        self.test_loader = DataLoader(
            SIDClassificationDataset(
                test_ds,
                image_size=cfg.data.image_size,
                normalize_mean=cfg.model.normalize_mean,
                normalize_std=cfg.model.normalize_std,
                transform=image_tf,
                transform_mask=mask_tf,
                joint_transform=joint_tf,
                return_mask=True,
            ),
            batch_size=cfg.loader.batch_size,
            shuffle=False,
            num_workers=cfg.loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.background_loader = None
        if eval_cfg and getattr(eval_cfg, "report_background", False):
            bg_samples = getattr(eval_cfg, "background_samples", 0) or cfg.data.test_samples
            background_ds = manager.get_split(
                split_type=TEST,
                max_samples=bg_samples,
                use_test_or_val_as_test_set=True,
                val_offset=cfg.data.val_samples,
                filter_fn=DatasetFilters.non_tampered,
            )
            self.background_loader = DataLoader(
                SIDClassificationDataset(
                    background_ds,
                    image_size=cfg.data.image_size,
                    normalize_mean=cfg.model.normalize_mean,
                    normalize_std=cfg.model.normalize_std,
                    transform=image_tf,
                    transform_mask=mask_tf,
                    joint_transform=joint_tf,
                    return_mask=False,
                ),
                batch_size=cfg.loader.batch_size,
                shuffle=False,
                num_workers=cfg.loader.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        # METRICS TRACKER
        # CAPTURE DICE/IOU RESULTS SO THEY CAN BE SAVED, PLOTTED, AND COMPARED ACROSS RUNS.
        self.metrics_tracker = EvaluationMetricsTracker(
            SegmentationEvaluationMetrics,
            logger=logger,
        )

        # MODEL
        # RESTORE THE BEST CHECKPOINTED WEIGHTS BEFORE RUNNING INFERENCE.
        self.model = TamperSegmentationModel(model_cfg=cfg.model, in_channels=3, out_channels=1).to(self.device)
        self.persister = persister or TorchModelPersister()
        self.persister.load_model(self.model, cfg.paths.model_path, device=self.device)
        self.logger.info(f"Loaded segmentation model from {cfg.paths.model_path}")

    def run(self):
        """Full evaluation pipeline."""
        self.logger.log_evaluation_config(asdict(self.cfg))
        # PIPELINE: RUN INFERENCE, SUMMARISE SCORES, OPTIONALLY EVALUATE BACKGROUND, THEN PLOT RESULTS.
        dice_scores, iou_scores, best_examples, worst_examples = self.infer()
        background_stats = None
        if self.background_loader is not None:
            background_stats = self.evaluate_background()
        self.compute_and_log(dice_scores, iou_scores, background_stats)
        self.save_and_plot(
            {
                "best_examples": best_examples,
                "worst_examples": worst_examples,
                "bucket_metrics": self.bucket_summary,
                "threshold_metrics": self.threshold_summary,
                "background_stats": background_stats,
            }
        )
        self.logger.info("Segmentation evaluation complete")

    @torch.no_grad()
    def infer(self):
        """Run inference and collect dice & iou per batch, plus gallery examples."""
        self.model.eval()
        dice_scores: List[float] = []
        iou_scores: List[float] = []
        self.bucket_summary = {label: {"dice": [], "iou": [], "count": 0} for label in self.bucket_labels}
        self.threshold_stats = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in self.thresholds}
        best_examples: List[Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        worst_examples: List[Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        zero_area_masks = 0

        mean_tensor = torch.tensor(self.cfg.model.normalize_mean, device=self.device).view(3, 1, 1)
        std_tensor = torch.tensor(self.cfg.model.normalize_std, device=self.device).view(3, 1, 1)

        self.logger.info("Starting segmentation evaluation")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Eval"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # THRESHOLD SWEEP: TRACK TRUE/POSITIVE/FALSE COUNTS FOR DIFFERENT PROBABILITY CUT-OFFS.
                for threshold in self.thresholds:
                    preds = (probs > threshold).float()
                    tp = (preds * masks).sum().item()
                    fp = (preds * (1 - masks)).sum().item()
                    fn = ((1 - preds) * masks).sum().item()
                    stats = self.threshold_stats[threshold]
                    stats["tp"] += tp
                    stats["fp"] += fp
                    stats["fn"] += fn

                pred_masks_default = (probs > 0.5).float()

                for idx in range(images.size(0)):
                    pred_mask = pred_masks_default[idx]
                    true_mask = masks[idx]
                    intersection = (pred_mask * true_mask).sum().item()
                    pred_sum = pred_mask.sum().item()
                    true_sum = true_mask.sum().item()
                    if true_sum <= EPS:
                        zero_area_masks += 1
                    dice_val = (2 * intersection + EPS) / (pred_sum + true_sum + EPS)
                    union = pred_sum + true_sum - intersection
                    iou_val = (intersection + EPS) / (union + EPS)

                    dice_scores.append(float(dice_val))
                    iou_scores.append(float(iou_val))

                    area_percent = float(true_sum / true_mask.numel() * 100.0)
                    bucket_label = self._assign_bucket(area_percent)
                    bucket_entry = self.bucket_summary[bucket_label]
                    bucket_entry["dice"].append(float(dice_val))
                    bucket_entry["iou"].append(float(iou_val))
                    bucket_entry["count"] += 1

                    # PREPARE DATA FOR QUALITATIVE REVIEW BY STORING IMAGE, TRUE MASK, AND PREDICTED MASK.
                    denorm = (images[idx] * std_tensor + mean_tensor).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    true_np = true_mask.squeeze(0).cpu().numpy()
                    pred_np = pred_mask.squeeze(0).cpu().numpy()
                    example = (denorm, true_np, pred_np)

                    self._add_ranked_example(best_examples, dice_val, example, reverse=True)
                    self._add_ranked_example(worst_examples, dice_val, example, reverse=False)

        self.bucket_summary = self._summarise_buckets(self.bucket_summary)
        self.threshold_summary = self._summarise_thresholds(self.threshold_stats)
        if zero_area_masks:
            self.logger.warning(
                f"Encountered {zero_area_masks} evaluation samples with zero-area tamper masks; "
                "consider inspecting dataset filters."
            )

        best_outputs = [data for _, data in best_examples]
        worst_outputs = [data for _, data in worst_examples]
        return dice_scores, iou_scores, best_outputs, worst_outputs

    def compute_and_log(self, dice_scores, iou_scores, background_stats=None):
        """Compute means, log, and record metrics."""
        mean_dice = float(np.mean(dice_scores)) if dice_scores else float("nan")
        mean_iou = float(np.mean(iou_scores)) if iou_scores else float("nan")
        self.logger.info(f"Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}")
        additional: Dict[str, Dict] = {}
        if background_stats:
            additional.update(background_stats)
        additional["bucket_metrics"] = self.bucket_summary
        additional["threshold_metrics"] = self.threshold_summary

        self.metrics_tracker.add_metrics(
            SegmentationEvaluationMetrics(
                task_type="segmentation",
                primary_metric="dice",
                primary_score=mean_dice,
                dice_coefficient=mean_dice,
                mean_iou=mean_iou,
                additional_metrics=additional,
            )
        )

    def save_and_plot(self, analysis: Dict[str, Dict]):
        """Persist metrics, save gallery, and generate plots."""
        metrics_path = self.cfg.paths.metrics_path
        self.metrics_tracker.save_to_json(metrics_path)
        self.logger.info(f"Saved evaluation metrics to {metrics_path}")

        plotter = SegmentationPlots(
            output_directory=self.cfg.paths.run_root,
            eval_history_path=metrics_path,
        )
        best_examples = analysis.get("best_examples") or []
        worst_examples = analysis.get("worst_examples") or []
        if best_examples:
            plotter.plot_segmentation_gallery(
                best_examples,
                ncols=3,
                filename="best_examples.png",
                save_path=self.cfg.paths.run_root,
                title_prefix="Best",
            )
        if worst_examples:
            plotter.plot_segmentation_gallery(
                worst_examples,
                ncols=3,
                filename="worst_examples.png",
                save_path=self.cfg.paths.run_root,
                title_prefix="Worst",
            )

        bucket_metrics = analysis.get("bucket_metrics")
        if bucket_metrics:
            plotter.plot_bucket_metrics(
                bucket_metrics,
                filename="dice_by_tamper_size.png",
                save_path=self.cfg.paths.run_root,
            )

        threshold_metrics = analysis.get("threshold_metrics")
        if threshold_metrics:
            plotter.plot_threshold_sweep(
                threshold_metrics,
                filename="threshold_sweep.png",
                save_path=self.cfg.paths.run_root,
            )

        background_stats = analysis.get("background_stats")
        if background_stats:
            plotter.plot_background_histograms(
                background_stats,
                filename="background_fp_hist.png",
                save_path=self.cfg.paths.run_root,
            )

        self.logger.info("Generated segmentation evaluation plots")

    def evaluate_background(self):
        """Measure false-positive behaviour on background-only images."""
        self.logger.info("Evaluating background false-positive rates")
        coverage = []
        max_probs = []
        with torch.no_grad():
            for batch in tqdm(self.background_loader, desc="Background Eval"):
                images = batch["image"].to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                coverage.extend(probs.mean(dim=(1, 2, 3)).cpu().tolist())
                max_probs.extend(probs.amax(dim=(1, 2, 3)).cpu().tolist())

        mean_cov = float(np.mean(coverage)) if coverage else float("nan")
        mean_max = float(np.mean(max_probs)) if max_probs else float("nan")
        over_half = int(np.sum(np.array(max_probs) > 0.5)) if max_probs else 0
        bins = np.linspace(0.0, 1.0, 21)
        cov_hist, _ = np.histogram(coverage, bins=bins)
        max_hist, _ = np.histogram(max_probs, bins=bins)
        self.logger.info(
            f"Background stats — mean probability: {mean_cov:.6f}, mean max probability: {mean_max:.6f}"
        )
        return {
            "background_mean_probability": mean_cov,
            "background_mean_max_probability": mean_max,
            "background_over_0.5": over_half,
            "background_samples": len(max_probs),
            "background_hist_bins": bins.tolist(),
            "background_hist_coverage": cov_hist.astype(int).tolist(),
            "background_hist_max": max_hist.astype(int).tolist(),
        }

    # HELPER UTILITIES
    def _build_bucket_labels(self, bounds: List[float]) -> List[str]:
        labels = []
        for idx in range(len(bounds) - 1):
            low = bounds[idx]
            high = bounds[idx + 1]
            if idx == 0:
                labels.append(f"<{self._format_percent(high)}")
            elif idx == len(bounds) - 2:
                labels.append(f">={self._format_percent(low)}")
            else:
                labels.append(f"{self._format_percent(low)}–{self._format_percent(high)}")
        return labels

    def _format_percent(self, value: float) -> str:
        if value == float("inf"):
            return "∞%"
        if value.is_integer():
            return f"{int(value)}%"
        return f"{value:.1f}%"

    def _assign_bucket(self, area_percent: float) -> str:
        for idx in range(len(self.bucket_bounds) - 1):
            low = self.bucket_bounds[idx]
            high = self.bucket_bounds[idx + 1]
            if low <= area_percent < high:
                return self.bucket_labels[idx]
        return self.bucket_labels[-1]

    def _add_ranked_example(
        self,
        container: List[Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
        score: float,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        reverse: bool,
    ) -> None:
        container.append((float(score), data))
        container.sort(key=lambda x: x[0], reverse=reverse)
        if len(container) > self.max_examples:
            container.pop(-1)

    def _summarise_buckets(self, bucket_raw: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for label, stats in bucket_raw.items():
            count = stats["count"]
            if count == 0:
                summary[label] = {"count": 0, "mean_dice": float("nan"), "mean_iou": float("nan")}
                continue
            summary[label] = {
                "count": count,
                "mean_dice": float(np.mean(stats["dice"])) if stats["dice"] else float("nan"),
                "mean_iou": float(np.mean(stats["iou"])) if stats["iou"] else float("nan"),
            }
        return summary

    def _summarise_thresholds(self, threshold_stats: Dict[float, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for threshold, stats in threshold_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            precision = tp / (tp + fp + EPS) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn + EPS) if (tp + fn) > 0 else float("nan")
            dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS) if (2 * tp + fp + fn) > 0 else float("nan")
            iou = (tp + EPS) / (tp + fp + fn + EPS) if (tp + fp + fn) > 0 else float("nan")
            summary[f"{threshold:.2f}"] = {
                "precision": float(precision),
                "recall": float(recall),
                "dice": float(dice),
                "iou": float(iou),
            }
        return summary
