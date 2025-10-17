#!/usr/bin/env python3
"""Command-line entry point for the Deepfake detection pipelines."""

import argparse
import os
import uuid
from pathlib import Path
from deepfake.utils.checkpoint_manager import CheckpointManager
import torch.multiprocessing as mp
from deepfake.classification import ClassificationTrainer, ClassificationEvaluator
from deepfake.segmentation import SegmentationTrainer, SegmentationEvaluator
from deepfake.visualization.classification_plots import ClassificationPlots
from deepfake.utils.training_metrics_tracker import TrainingMetricsTracker
from deepfake.visualization.segmentation_plots    import SegmentationPlots
from deepfake.utils.seed_manager import SeedManager
from deepfake.utils.logger       import SidLogger
from deepfake.utils.config_loader import ConfigLoader
from deepfake.config import Config
from deepfake.utils.tuner import run_tuning, DEFAULT_STORAGE


def make_parser():
    """CREATE ARGUMENT PARSER AND DEFINE SUBCOMMANDS."""
    parser = argparse.ArgumentParser(
        prog="deepfake",
        description="Deepfake Detection Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared options
    def add_common_args(p):
        """ADD COMMON TASK/ENV/ID/CHECKPOINT ARGS TO SUBPARSER."""
        p.add_argument("--task",   choices=["classification","segmentation"], required=True)
        p.add_argument("--env",    choices=["dev","test"], default="dev")
        p.add_argument("--runid",  type=str, help="Run ID (for eval/plot)")
        p.add_argument("--checkpoint", type=int,
                       help="Epoch number to resume training from checkpoint")

    # train subcommand
    train_p = subparsers.add_parser("train", help="Train a pipeline")
    add_common_args(train_p)

    # eval subcommand
    eval_p = subparsers.add_parser("eval", help="Evaluate a pipeline")
    add_common_args(eval_p)

    # plot subcommand
    plot_p = subparsers.add_parser("plot", help="Plot metrics")
    add_common_args(plot_p)
    plot_p.add_argument(
        "--stage",
        choices=["train","eval"],
        required=True,
        help="Which metrics to plot",
    )

    tune_p = subparsers.add_parser("tune", help="Run hyperparameter search")
    tune_p.add_argument("--task", choices=["classification", "segmentation"], required=True)
    tune_p.add_argument("--env", choices=["dev", "test"], default="dev")
    tune_p.add_argument("--n-trials", type=int, default=20)
    tune_p.add_argument("--timeout", type=int, default=None)
    tune_p.add_argument("--study-name", default=None)
    tune_p.add_argument("--storage", default=DEFAULT_STORAGE)
    tune_p.add_argument("--pruner", action="store_true")
    tune_p.add_argument("--resume", action="store_true")
    tune_p.add_argument("--keep-runs", action="store_true")
    tune_p.add_argument("--device", default=None)

    return parser

def build_config_path(task: str, env: str) -> str:
    """DETERMINE CONFIG FILE PATH BASED ON TASK AND ENV."""
    prefix = f"{env}-" if env == "dev" else ""
    return f"configs/{prefix}{task}.yaml"

def run_train(cfg: Config, args):
    """SET UP LOGGER, METRICS, CHECKPOINT MANAGER, AND DISPATCH TRAINER."""
    logger = SidLogger(name=f"{args.task}-train", log_dir=cfg.paths.log_path)
    metrics_tracker = TrainingMetricsTracker(logger=logger.logger)
    metrics_tracker.configure_backup(cfg.paths.history_path)
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoints_path,
        logger=logger.logger,
        backup_frequency=cfg.training.checkpoint_frequency,
        keep_checkpoints=cfg.training.keep_checkpoints,
    )

    # determine start_epoch
    start_epoch = 0
    resume_epoch = None
    if args.checkpoint is not None:
        resume_epoch = args.checkpoint
        start_epoch = resume_epoch
        logger.info(f"Resuming {cfg.paths.run_id} from checkpoint epoch {args.checkpoint}")
    elif os.path.isfile(cfg.paths.history_path):
        metrics_tracker.load_from_json(cfg.paths.history_path)
        logger.info(f"Loaded history for run {cfg.paths.run_id}")

    if args.task == cfg.Task.CLASSIFICATION:
        trainer = ClassificationTrainer(
            cfg=cfg,
            logger=logger,
            metrics_tracker=metrics_tracker,
            checkpoint_mgr=checkpoint_mgr,
        )
    else:
        trainer = SegmentationTrainer(
            cfg=cfg,
            logger=logger,
            metrics_tracker=metrics_tracker,
            checkpoint_mgr=checkpoint_mgr,
        )

    if resume_epoch is not None:
        checkpoint_data = checkpoint_mgr.load_checkpoint(
            resume_epoch,
            model=trainer.model,
            optimizer=trainer.optimizer,
        )
        if checkpoint_data.get('metrics_path'):
            metrics_tracker.load_from_json(checkpoint_data['metrics_path'])
            logger.info(f"Loaded metrics from checkpoint epoch {resume_epoch}")
        if not checkpoint_data.get('model_loaded'):
            logger.warning(f"Checkpoint epoch {resume_epoch} missing model state; continuing with current weights")
        if not checkpoint_data.get('optimizer_loaded'):
            logger.warning(f"Checkpoint epoch {resume_epoch} missing optimizer state; optimizer reinitialised")
        if getattr(trainer, 'load_ema_state', None):
            checkpoint_dir = checkpoint_data.get('checkpoint_dir')
            if checkpoint_dir is not None:
                trainer.load_ema_state(Path(checkpoint_dir) / 'ema_state.pth')

    trainer.train(start_epoch=start_epoch)

def run_eval(cfg: Config, args):
    """Dispatch evaluation based on task, defaulting to the latest trained run when --runid is omitted."""
    initial_run_root = cfg.paths.run_root

    if args.runid:
        cfg.paths.run_id = args.runid
    else:
        runs_root = initial_run_root.parent
        if not runs_root.exists():
            raise FileNotFoundError(
                f"No runs found under {runs_root}. Provide --runid to evaluate a specific run."
            )

        candidates = sorted(
            (p for p in runs_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        model_filename = cfg.paths.model_filename
        selected = next((p for p in candidates if (p / model_filename).exists()), None)
        if selected is None:
            raise FileNotFoundError(
                f"Could not find any run in {runs_root} containing {model_filename}."
            )

        cfg.paths.run_id = selected.name

    new_run_root = cfg.paths.run_root
    if initial_run_root != new_run_root and initial_run_root.exists():
        try:
            initial_run_root.rmdir()
        except OSError:
            pass

    logger = SidLogger(name=f"{args.task}-eval", log_dir=cfg.paths.log_path)
    if args.runid is None:
        logger.info(
            f"--runid not provided; evaluating latest run {cfg.paths.run_id} from {cfg.paths.run_root}"
        )

    if args.task == cfg.Task.CLASSIFICATION:
        ClassificationEvaluator(cfg, logger).run()
    elif args.task == cfg.Task.SEGMENTATION:
        SegmentationEvaluator(cfg, logger).run()


def _resolve_run_for_stage(cfg: Config, required_file: str) -> Path:
    initial_run_root = cfg.paths.run_root
    if cfg.paths.run_id and (initial_run_root / required_file).exists():
        return initial_run_root

    runs_root = initial_run_root.parent
    if not runs_root.exists():
        raise FileNotFoundError(
            f"No runs found under {runs_root}. Provide --runid to plot a specific run."
        )

    for candidate in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        if (candidate / required_file).exists():
            cfg.paths.run_id = candidate.name
            try:
                initial_run_root.rmdir()
            except OSError:
                pass
            return candidate

    raise FileNotFoundError(
        f"Could not find any run in {runs_root} containing {required_file}."
    )


def run_plot(cfg: Config, args):
    """Dispatch plotting based on task and stage, defaulting to the latest run when --runid is omitted."""
    if args.runid:
        cfg.paths.run_id = args.runid

    required = cfg.paths.history_filename if args.stage == "train" else cfg.paths.metrics_filename
    run_root = _resolve_run_for_stage(cfg, required)

    if args.runid is None:
        print(f"Plotting stage '{args.stage}' for latest run {cfg.paths.run_id}")

    if args.task == cfg.Task.CLASSIFICATION:
        plots = ClassificationPlots(
            training_history_path=cfg.paths.history_path if args.stage=="train" else None,
            eval_history_path=   cfg.paths.metrics_path  if args.stage=="eval"  else None,
            output_directory=run_root
        )
        if args.stage == "train":
            plots.plot_training_history()
            plots.plot_learning_rate_schedule()
        else:
            plots.plot_confusion_matrix()
            plots.plot_classification_report()

    else:
        plots = SegmentationPlots(
            training_history_path=cfg.paths.history_path if args.stage=="train" else None,
            eval_history_path=   cfg.paths.metrics_path  if args.stage=="eval"  else None,
            output_directory=run_root
        )
        if args.stage == "train":
            plots.plot_training_history()
        else:
            print("Segmentation eval plotting not implemented yet.")


def run_tune(args):
    """Execute an Optuna-based hyperparameter sweep."""
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


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "tune":
        run_tune(args)
        return

    # Load config
    cfg_path = build_config_path(args.task, args.env)
    cfg : Config = ConfigLoader(config_path=cfg_path).get_config()
    
    # To Override run_id from command line, so we can evaluate a model
    if args.runid:
        # Remove the new directory created by default
        cfg.paths.run_root.rmdir()
        cfg.paths.run_id = args.runid
        
    # Seed
    seed = getattr(cfg.training, "seed", 42)
    SeedManager(seed)

    # Dispatch
    if args.command == "train":
        run_train(cfg, args)
    elif args.command == "eval":
        run_eval(cfg, args)
    elif args.command == "plot":
        run_plot(cfg, args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def entrypoint():
    """ENSURE MULTIPROCESSING USES SPAWN AND LAUNCH MAIN."""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
        
    main()


if __name__ == "__main__":
    entrypoint()
