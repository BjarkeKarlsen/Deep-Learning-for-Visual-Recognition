"""Segmentation pipeline modules."""

from .train import Trainer as SegmentationTrainer
from .evaluate import Evaluator as SegmentationEvaluator

__all__ = ["train", "evaluate"]
