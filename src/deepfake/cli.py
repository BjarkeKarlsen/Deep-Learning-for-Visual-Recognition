#!/usr/bin/env python3
"""Command-line entry point for the Deepfake detection pipelines."""

import argparse
import os
import uuid
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
            logger.logger.warning(f"Checkpoint epoch {resume_epoch} missing model state; continuing with current weights")
        if not checkpoint_data.get('optimizer_loaded'):
            logger.logger.warning(f"Checkpoint epoch {resume_epoch} missing optimizer state; optimizer reinitialised")

    trainer.train(start_epoch=start_epoch)

def run_eval(cfg : Config, args):
    """DISPATCH EVALUATION BASED ON TASK."""
    logger = SidLogger(name=f"{args.task}-eval", log_dir=cfg.paths.log_path)
    if args.task == cfg.Task.CLASSIFICATION:
        ClassificationEvaluator(cfg, logger).run()
    elif args.task == cfg.Task.SEGMENTATION:
        SegmentationEvaluator(cfg, logger).run()


def run_plot(cfg: Config, args):
    """DISPATCH PLOTTING BASED ON TASK AND STAGE."""
    run_root = cfg.paths.run_root
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


def main():
    parser = make_parser()
    args = parser.parse_args()
    
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


def entrypoint():
    """ENSURE MULTIPROCESSING USES SPAWN AND LAUNCH MAIN."""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
        
    main()


if __name__ == "__main__":
    entrypoint()
