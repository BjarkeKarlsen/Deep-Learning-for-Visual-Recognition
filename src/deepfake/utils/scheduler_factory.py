from __future__ import annotations

from typing import Optional, Tuple

import torch

from deepfake.config.schema import TrainingConfig


class SchedulerFactory:
    """Build learning rate schedulers based on the training configuration."""

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer,
        training_cfg: TrainingConfig,
        *,
        steps_per_epoch: int,
    ) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], str]:
        """Return a scheduler and step granularity ("epoch" or "batch")."""
        name = (training_cfg.scheduler.name or "").lower()
        # EXIT EARLY WHEN NO SCHEDULER IS REQUESTED.
        if not name or name == "none":
            return None, "epoch"

        if name == "cosine":
            # STANDARD COSINE ANNEALING WITH OPTIONAL USER T_MAX.
            t_max = training_cfg.scheduler.t_max or max(training_cfg.epochs, 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
            )
            return scheduler, "epoch"

        if name == "onecycle":
            if steps_per_epoch <= 0:
                raise ValueError("steps_per_epoch must be positive for OneCycleLR")
            # ONE CYCLE LR THAT STEPS EACH BATCH FOR SMOOTH ANNEALING.
            max_lr = training_cfg.scheduler.max_lr or training_cfg.learning_rate
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=training_cfg.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=training_cfg.scheduler.pct_start,
                div_factor=training_cfg.scheduler.div_factor,
                final_div_factor=training_cfg.scheduler.final_div_factor,
            )
            return scheduler, "batch"

        raise ValueError(f"Unsupported scheduler '{training_cfg.scheduler.name}'")
