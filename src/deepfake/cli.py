#!/usr/bin/env python3
"""Command-line entry point for the Deepfake detection pipelines."""

import argparse
import torch.multiprocessing as mp
from deepfake.classification import train as classify_train, evaluate as classify_eval
from deepfake.segmentation    import train as segment_train, evaluate as segment_eval
from deepfake.visualization.classification_plots import ClassificationPlots
from deepfake.visualization.segmentation_plots    import SegmentationPlots
from deepfake.utils.seed_manager import SeedManager
from deepfake.utils.logger       import SidLogger
from deepfake.utils.config_loader import ConfigLoader
from deepfake.config import Config


def make_parser():
    parser = argparse.ArgumentParser(
        prog="deepfake",
        description="Deepfake Detection Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared options
    def add_common_args(p):
        p.add_argument("--task",   choices=["classification","segmentation"], required=True)
        p.add_argument("--env",    choices=["dev","test"], default="dev")
        p.add_argument("--runid",  type=str, help="Run ID (for eval/plot)")

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
    prefix = f"{env}-" if env == "dev" else ""
    return f"configs/{prefix}{task}.yaml"

def run_train(cfg : Config, args):
    logger = SidLogger(name=f"{args.task}-train", log_dir=cfg.paths.log_path)
    if args.task == cfg.task.CLASSIFICATION:
        classify_train(logger, cfg)
    else:
        segment_train(logger, cfg)


def run_eval(cfg : Config, args):
    logger = SidLogger(name=f"{args.task}-eval", log_dir=cfg.paths.log_path)
    if args.task == cfg.task.CLASSIFICATION:
        classify_eval(logger, cfg)
    elif args.task == cfg.task.SEGMENTATION:
        segment_eval(logger, cfg)


def run_plot(cfg: Config, args):
    run_root = cfg.paths.run_root
    if args.task == cfg.task.CLASSIFICATION:
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
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    main()


if __name__ == "__main__":
    entrypoint()
