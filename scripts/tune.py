#!/usr/bin/env python3
"""Wrapper around `deepfake-cli tune` for backwards compatibility."""

from __future__ import annotations

import argparse

from deepfake.utils.tuner import run_tuning, DEFAULT_STORAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for the Deepfake toolkit")
    parser.add_argument("--task", choices=["classification", "segmentation"], required=True)
    parser.add_argument("--env", choices=["dev", "test"], default="dev")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--pruner", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = run_tuning(
        task=args.task,
        env=args.env,
        n_trials=args.n_trials,
        timeout=args.timeout,
        storage=args.storage,
        study_name=args.study_name,
        pruner=args.pruner,
        resume=args.resume,
        keep_runs=args.keep_runs,
        device=args.device,
    )
    best = study.best_trial
    print("Best trial:")
    print(f"  value: {best.value}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
