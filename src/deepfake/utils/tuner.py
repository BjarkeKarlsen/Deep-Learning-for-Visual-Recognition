"""Optuna-based hyperparameter tuning utilities."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

try:
    import optuna
    from optuna.trial import Trial
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore
    Trial = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
from omegaconf import OmegaConf

from deepfake.config import Config as ConfigSchema

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_STORAGE = "sqlite:///optuna_study.db"


def _build_base_config_path(task: str, env: str) -> Path:
    # LOCATE A CONFIG TEMPLATE THAT MATCHES THE REQUESTED TASK AND ENV.
    candidate_names = [
        f"{env}-{task}.yaml",
        f"{env}-{task}-smoke.yaml",
        f"{task}-{env}.yaml",
        f"{task}.yaml",
    ]
    for name in candidate_names:
        candidate = CONFIGS_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No base config found for task='{task}' env='{env}'. Checked: {candidate_names}"
    )


def _load_best_metric(run_root: Path, task: str) -> float:
    # EXTRACT THE BEST VALIDATION SCORE FROM A COMPLETED RUN DIRECTORY.
    history_path = run_root / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Did not find history.json in {run_root}")
    import json

    with history_path.open() as f:
        payload = json.load(f)
    metrics = payload.get("metrics", [])
    if not metrics:
        raise ValueError("history.json contained no metrics entries")

    if task == "classification":
        best_val = max((m.get("val_acc") for m in metrics if m.get("val_acc") is not None), default=None)
    else:
        best_val = max(
            (
                m.get("additional_metrics", {}).get("val_dice")
                for m in metrics
                if m.get("additional_metrics")
            ),
            default=None,
        )
    if best_val is None:
        raise ValueError("Could not extract validation metric from history.json")
    return float(best_val)


def _sample_params_classification(trial: Trial) -> Dict[str, Any]:
    # DEFINE SEARCH SPACE FOR CLASSIFICATION EXPERIMENTS.
    optimizer_name = trial.suggest_categorical("clf_optimizer", ["adam", "adamw"])
    scheduler_name = trial.suggest_categorical("clf_scheduler", ["", "cosine", "onecycle"])
    augment_enabled = trial.suggest_categorical("clf_aug_enable", [True, False])

    scheduler_cfg: Dict[str, Any] = {"name": scheduler_name}
    if scheduler_name == "onecycle":
        scheduler_cfg.update(
            {
                "max_lr": trial.suggest_float("clf_max_lr", 5e-4, 5e-3, log=True),
                "pct_start": trial.suggest_float("clf_pct_start", 0.05, 0.35),
                "div_factor": trial.suggest_float("clf_div_factor", 5.0, 30.0),
                "final_div_factor": 10000.0,
            }
        )
    elif scheduler_name == "cosine":
        scheduler_cfg.update(
            {
                "t_max": trial.suggest_int("clf_cosine_t_max", 5, 40),
            }
        )

    return {
        "training": {
            "learning_rate": trial.suggest_float("clf_base_lr", 5e-5, 5e-3, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.08),
            "ema_decay": trial.suggest_float("ema_decay", 0.0, 0.9999),
            "grad_clip_norm": trial.suggest_float("grad_clip", 0.5, 4.0),
            "optimizer": {
                "name": optimizer_name,
                "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True),
            },
            "scheduler": scheduler_cfg,
        },
        "data": {
            "augment": {
                "enable": augment_enabled,
                "random_resized_crop": trial.suggest_categorical("clf_rrc", [True, False]),
                "horizontal_flip_prob": trial.suggest_float("clf_hflip", 0.0, 0.7),
                "color_jitter_brightness": trial.suggest_float("clf_cj_bright", 0.0, 0.3),
                "color_jitter_contrast": trial.suggest_float("clf_cj_contrast", 0.0, 0.3),
                "color_jitter_saturation": trial.suggest_float("clf_cj_sat", 0.0, 0.3),
                "color_jitter_hue": trial.suggest_float("clf_cj_hue", 0.0, 0.05),
                "gaussian_blur_prob": trial.suggest_float("clf_blur_prob", 0.0, 0.4),
                "random_erasing_prob": trial.suggest_float("clf_erasing_prob", 0.0, 0.4),
            }
        },
    }


def _sample_params_segmentation(trial: Trial) -> Dict[str, Any]:
    # DEFINE SEARCH SPACE FOR SEGMENTATION EXPERIMENTS.
    scheduler_name = trial.suggest_categorical("seg_scheduler", ["onecycle", "cosine"])
    scheduler_cfg: Dict[str, Any] = {"name": scheduler_name}
    if scheduler_name == "onecycle":
        scheduler_cfg.update(
            {
                "max_lr": trial.suggest_float("seg_max_lr", 1e-3, 6e-3, log=True),
                "pct_start": trial.suggest_float("seg_pct_start", 0.05, 0.4),
                "div_factor": trial.suggest_float("seg_div_factor", 5.0, 30.0),
                "final_div_factor": 10000.0,
            }
        )
    else:
        scheduler_cfg.update(
            {
                "t_max": trial.suggest_int("seg_cosine_t_max", 10, 60),
            }
        )

    loss_type = trial.suggest_categorical("seg_loss_type", ["bce", "focal"])
    primary_weight = trial.suggest_float("seg_primary_weight", 0.3, 0.8)
    loss_cfg: Dict[str, Any] = {
        "type": loss_type,
        "bce_weight": primary_weight,
        "dice_weight": 1.0 - primary_weight,
    }
    if loss_type == "focal":
        loss_cfg.update(
            {
                "focal_alpha": trial.suggest_float("seg_focal_alpha", 0.1, 0.6),
                "focal_gamma": trial.suggest_float("seg_focal_gamma", 1.0, 5.0),
            }
        )

    augment_enabled = trial.suggest_categorical("seg_aug_enable", [True, False])
    scale_min = trial.suggest_float("scale_min", 0.4, 0.9)
    scale_delta = trial.suggest_float("scale_delta", 0.05, 0.4)
    scale_max = min(1.3, scale_min + scale_delta)
    flip_prob = trial.suggest_float("horizontal_flip_prob", 0.0, 0.9)
    blur_prob = trial.suggest_float("blur_prob", 0.0, 0.4)
    blur_sigma_min = trial.suggest_float("blur_sigma_min", 0.03, 0.2)
    blur_sigma_max = trial.suggest_float("blur_sigma_max", 0.5, 2.5)
    if blur_sigma_max <= blur_sigma_min:
        blur_sigma_max = blur_sigma_min + 0.1

    cj_brightness = trial.suggest_float("color_jitter_brightness", 0.0, 0.35)
    cj_contrast = trial.suggest_float("color_jitter_contrast", 0.0, 0.35)
    cj_saturation = trial.suggest_float("color_jitter_saturation", 0.0, 0.3)
    cj_hue = trial.suggest_float("color_jitter_hue", 0.0, 0.08)

    re_prob = trial.suggest_float("random_erasing_prob", 0.0, 0.25)
    re_scale_min = trial.suggest_float("random_erasing_scale_min", 0.002, 0.05)
    re_scale_max = trial.suggest_float("random_erasing_scale_max", 0.05, 0.45)
    if re_scale_max <= re_scale_min:
        re_scale_max = re_scale_min + 0.01
    re_ratio_min = trial.suggest_float("random_erasing_ratio_min", 0.1, 0.7)
    re_ratio_max = trial.suggest_float("random_erasing_ratio_max", 1.5, 5.0)
    if re_ratio_max <= re_ratio_min:
        re_ratio_max = re_ratio_min + 0.5

    return {
        "training": {
            "learning_rate": trial.suggest_float("seg_base_lr", 3e-4, 3e-3, log=True),
            "grad_clip_norm": trial.suggest_float("seg_grad_clip", 0.5, 3.0),
            "ema_decay": trial.suggest_float("seg_ema_decay", 0.0, 0.999),
            "optimizer": {
                "weight_decay": trial.suggest_float("seg_weight_decay", 1e-5, 5e-3, log=True),
            },
            "scheduler": scheduler_cfg,
            "loss": loss_cfg,
        },
        "loader": {
            "batch_size": trial.suggest_categorical("seg_batch_size", [8, 12, 16, 20, 24, 28, 32]),
        },
        "data": {
            "augment": {
                "enable": augment_enabled,
                "gaussian_blur_prob": blur_prob,
                "gaussian_blur_sigma_min": blur_sigma_min,
                "gaussian_blur_sigma_max": blur_sigma_max,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "horizontal_flip_prob": flip_prob,
                "color_jitter_brightness": cj_brightness,
                "color_jitter_contrast": cj_contrast,
                "color_jitter_saturation": cj_saturation,
                "color_jitter_hue": cj_hue,
                "random_erasing_prob": re_prob,
                "random_erasing_scale_min": re_scale_min,
                "random_erasing_scale_max": re_scale_max,
                "random_erasing_ratio_min": re_ratio_min,
                "random_erasing_ratio_max": re_ratio_max,
            }
        },
    }


def _build_config(base_cfg_path: Path, overrides: Dict[str, Any], device_override: Optional[str]) -> ConfigSchema:
    # MERGE BASE CONFIG WITH TRIAL OVERRIDES AND OPTIONAL DEVICE HINT.
    base_struct = OmegaConf.structured(ConfigSchema)
    base_cfg = OmegaConf.load(base_cfg_path)
    override_conf = OmegaConf.create(overrides)
    merged = OmegaConf.merge(base_struct, base_cfg, override_conf)
    if device_override:
        merged.training.device = device_override
    cfg_obj: ConfigSchema = OmegaConf.to_object(merged)
    return cfg_obj


def run_tuning(
    *,
    task: str,
    env: str,
    n_trials: int,
    timeout: Optional[int],
    storage: str = DEFAULT_STORAGE,
    study_name: Optional[str] = None,
    pruner: bool = False,
    resume: bool = False,
    keep_runs: bool = False,
    device: Optional[str] = None,
) -> optuna.study.Study:
    if optuna is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "Optuna is not installed. Install it with `pip install optuna` to use the tune command."
        ) from _IMPORT_ERROR
    base_cfg_path = _build_base_config_path(task, env)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=resume,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner() if pruner else optuna.pruners.NopPruner(),
    )

    def objective(trial: Trial) -> float:
        # TRAIN A SINGLE TRIAL AND REPORT THE VALIDATION METRIC TO OPTUNA.
        overrides = (
            _sample_params_classification(trial)
            if task == "classification"
            else _sample_params_segmentation(trial)
        )
        cfg = _build_config(base_cfg_path, overrides, device)

        default_root = cfg.paths.run_root
        if default_root.exists():
            shutil.rmtree(default_root, ignore_errors=True)

        run_id = time.strftime("%Y%m%dT%H%M%S") + f"_trial{trial.number}"
        cfg.paths.run_id = run_id
        run_root = cfg.paths.run_root
        run_root.mkdir(parents=True, exist_ok=True)

        trial.set_user_attr("overrides", overrides)

        train_args = SimpleNamespace(task=task, env=env, runid=None, checkpoint=None)
        try:
            from deepfake.cli import run_train as _run_train  # local import to avoid circular dependencies
            _run_train(cfg, train_args)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            if not keep_runs:
                shutil.rmtree(run_root, ignore_errors=True)
            raise

        metric = _load_best_metric(run_root, task)
        if not keep_runs:
            shutil.rmtree(run_root, ignore_errors=True)
        return metric

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


__all__ = ["run_tuning", "DEFAULT_STORAGE"]
