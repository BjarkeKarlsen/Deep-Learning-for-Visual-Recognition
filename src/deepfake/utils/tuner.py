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
    return {
        "training": {
            "learning_rate": trial.suggest_float("base_lr", 5e-5, 5e-4, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.08),
            "ema_decay": trial.suggest_float("ema_decay", 0.9, 0.9999),
            "grad_clip_norm": trial.suggest_float("grad_clip", 0.5, 2.0),
            "optimizer": {
                "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True),
            },
            "scheduler": {
                "max_lr": trial.suggest_float("max_lr", 1e-3, 5e-3, log=True),
                "pct_start": trial.suggest_float("pct_start", 0.1, 0.4),
                "div_factor": trial.suggest_float("div_factor", 10.0, 30.0),
            },
        },
        "data": {
            "augment": {
                "gaussian_blur_prob": trial.suggest_float("blur_prob", 0.0, 0.3),
                "random_erasing_prob": trial.suggest_float("erasing_prob", 0.0, 0.3),
            }
        },
    }


def _sample_params_segmentation(trial: Trial) -> Dict[str, Any]:
    # DEFINE SEARCH SPACE FOR SEGMENTATION EXPERIMENTS.
    return {
        "training": {
            "learning_rate": trial.suggest_float("base_lr", 5e-4, 2e-3, log=True),
            "grad_clip_norm": trial.suggest_float("grad_clip", 0.5, 2.0),
            "optimizer": {
                "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True),
            },
            "scheduler": {
                "max_lr": trial.suggest_float("max_lr", 2e-3, 6e-3, log=True),
                "pct_start": trial.suggest_float("pct_start", 0.1, 0.4),
                "div_factor": trial.suggest_float("div_factor", 10.0, 30.0),
            },
        },
        "data": {
            "augment": {
                "gaussian_blur_prob": trial.suggest_float("blur_prob", 0.0, 0.3),
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
