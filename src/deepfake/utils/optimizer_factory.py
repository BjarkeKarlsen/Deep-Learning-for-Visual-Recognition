from typing import Iterable

import torch
from torch.optim import Optimizer

from deepfake.config.schema import OptimizerConfig, TrainingConfig


class OptimizerFactory:
    def __new__(
        cls,
        parameters: Iterable[torch.nn.Parameter],
        training_cfg: TrainingConfig,
    ) -> Optimizer:
        """
        Construct a torch.optim.Optimizer respecting the configuration.
    
        Supports `adam`, `adamw`, and `sgd`.
        """
        opt_cfg = training_cfg.optimizer
        name = (opt_cfg.name or "adam").lower()
        lr = training_cfg.learning_rate
        weight_decay = opt_cfg.weight_decay

        if name == "adam":
            betas = cls._resolve_betas(opt_cfg)
            return torch.optim.Adam(
                parameters,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        if name == "adamw":
            betas = cls._resolve_betas(opt_cfg)
            return torch.optim.AdamW(
                parameters,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=opt_cfg.momentum,
                weight_decay=weight_decay,
                nesterov=opt_cfg.nesterov,
            )

        raise ValueError(
            f"Unsupported optimizer '{opt_cfg.name}'. "
            "Expected one of: 'adam', 'adamw', 'sgd'."
        )
        
    @staticmethod
    def _resolve_betas(opt_cfg: OptimizerConfig) -> tuple[float, float]:
        betas = getattr(opt_cfg, "betas", None) or [0.9, 0.999]
        if len(betas) != 2:
            raise ValueError(f"Optimizer betas must contain exactly 2 values (got {len(betas)})")
        return float(betas[0]), float(betas[1])

