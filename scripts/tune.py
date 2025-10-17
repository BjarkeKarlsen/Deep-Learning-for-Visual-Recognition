#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning harness for deepfake classification/segmentation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import optuna
from optuna.trial import Trial
from omegaconf import OmegaConf

from deepfake.cli import run_train
from deepfake.config import Config as ConfigSchema

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "deepfake"
CONFIGS_DIR = ROOT / "configs"
DEFAULT_STORAGE = "sqlite:///optuna_study.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for the Deepfake toolkit")
    parser.add_argument("--task", choices=["classification", "segmentation"], required=True)
    parser.add_argument("--env", default="dev", help="Config environment (dev/test) to base overrides on")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Global study timeout in seconds")
    parser.add_argument("--study-name", default=None, help="Optuna study name")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, help="Optuna storage URL")
    parser.add_argument("--pruner", action="store_true", help="Enable median pruner")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--keep-runs", action="store_true", help="Preserve outputs/ runs for manual inspection")
    parser.add_argument("--device", default=None, help="Override training.device (cuda/cpu)")
    return parser.parse_args()


def build_base_config(task: str, env: str) -> Path:
    cfg_name = f"{env}-{task}.yaml" if (CONFIGS_DIR / f"{env}-{task}.yaml").exists() else f"{task}.yaml"
    return CONFIGS_DIR / cfg_name


def load_best_metric(run_root: Path, task: str) -> float:
    history_path = run_root / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Did not find history.json in {run_root}")
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


def sample_params_classification(trial: Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {
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
        }
    }
    return params


def sample_params_segmentation(trial: Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {
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
        }
    }
    return params


def build_config(base_cfg: Path, overrides: Dict[str, Any], device_override: Optional[str]) -> ConfigSchema:
    """
    Merge overrides into the base configuration and return a Config dataclass.
    """
    base_struct = OmegaConf.structured(ConfigSchema)
    user_conf = OmegaConf.load(base_cfg)
    override_conf = OmegaConf.create(overrides)
    merged = OmegaConf.merge(base_struct, user_conf, override_conf)
    if device_override:
        merged.training.device = device_override
    cfg_obj: ConfigSchema = OmegaConf.to_object(merged)
    return cfg_obj


def objective(args: argparse.Namespace, base_cfg: Path, trial: Trial) -> float:
    overrides = sample_params_classification(trial) if args.task == "classification" else sample_params_segmentation(trial)
    cfg = build_config(base_cfg, overrides, args.device)
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"_trial{trial.number}"
    # Remove default run directory created during config initialisation
    default_root = cfg.paths.run_root
    if default_root.exists():
        shutil.rmtree(default_root, ignore_errors=True)

    cfg.paths.run_id = run_id
    run_root = cfg.paths.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    trial.set_user_attr("overrides", overrides)

    train_args = SimpleNamespace(task=args.task, env=args.env, runid=None, checkpoint=None)

    try:
        run_train(cfg, train_args)
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        if not args.keep_runs:
            shutil.rmtree(run_root, ignore_errors=True)
        raise

    metric = load_best_metric(run_root, args.task)
    if not args.keep_runs:
        shutil.rmtree(run_root, ignore_errors=True)
    return metric


def main() -> None:
    args = parse_args()
    base_cfg = build_base_config(args.task, args.env)
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner(),
    )
    study.optimize(lambda t: objective(args, base_cfg, t), n_trials=args.n_trials, timeout=args.timeout)

    print("Best trial:")
    best = study.best_trial
    print(f"  value: {best.value}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
