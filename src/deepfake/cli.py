#!/usr/bin/env python3
"""Command-line entry point for the Deepfake detection pipelines."""

import argparse
import torch.multiprocessing as mp

from deepfake.classification import evaluate as classify_evaluate
from deepfake.classification import train as classify_train
from deepfake.segmentation import evaluate as segment_evaluate
from deepfake.segmentation import train as segment_train
from deepfake.utils.seed_manager import SeedManager
from deepfake.utils.logger import SidLogger
from deepfake.utils.config_loader import ConfigLoader
from deepfake.visualization.classification_plots import ClassificationPlots


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake Detection Toolkit")
    parser.add_argument("--train", action="store_true", help="Train the selected task pipeline")
    parser.add_argument("--eval", action="store_true", help="Evaluate the selected task pipeline")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        help="Pipeline to execute",
    )
    
    parser.add_argument(
        '--plot',
        type=str,
        choices=["classification", "segmentation"],
        help='Create plots from training history JSON file.'
    )
        
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML configuration override",
    )
    args = parser.parse_args()

    if args.train and not args.eval:
        run_mode = "train"
    elif args.eval and not args.train:
        run_mode = "eval"
    else:
        run_mode = None

    loader = ConfigLoader(config_path=args.config, run_mode=run_mode)
    cfg = loader.get_config()

    seed_value = getattr(cfg.training, "seed", None) or 42
    SeedManager(seed_value)

    log_dir = cfg.paths.logging_dir or "outputs/logs"

    if args.task == cfg.Task.CLASSIFICATION:
        if args.train:
            classify_train(SidLogger("classification-train", log_dir=log_dir), cfg)
        if args.eval:
            classify_evaluate(SidLogger("classification-eval", log_dir=log_dir), cfg)
    elif args.task == cfg.Task.SEGMENTATION:
        if args.train:
            segment_train(SidLogger("segmentation-train", log_dir=log_dir), cfg)
        if args.eval:
            segment_evaluate(SidLogger("segmentation-eval", log_dir=log_dir), cfg)
    elif args.plot:
        # Use existing ClassificationPlots utilities as in main; behavior is unchanged.
        if args.plot == cfg.Task.CLASSIFICATION:
            class_plots = ClassificationPlots(
                history_path="outputs/results/dev/classification/training_history.json",
                save_path=cfg.paths.results_dir,
                eval_report_path="outputs/results/classification/evaluation.json",
            )

            if args.eval:
                class_plots.plot_confusion_matrix()
                class_plots.plot_classification_report()
            elif args.train:
                class_plots.plot_training_history()
                class_plots.plot_learning_rate_schedule()
        elif args.plot == cfg.Task.SEGMENTATION:
            print("Plotting segmentation metrics...")
            # Placeholder for actual plotting function
            # plot_segmentation_metrics(cfg.paths.history_file, cfg.paths.results_dir)
    else:
        print(f"Unknown task: {args.task}")
        parser.print_help()
        return



def entrypoint() -> None:
    """Enforce the 'spawn' start method before delegating to the CLI main routine."""
    start_method = mp.get_start_method(allow_none=True)
    if start_method != "spawn":
        mp.set_start_method(
            "spawn",
            force=True,
        )
    main()


if __name__ == "__main__":
    entrypoint()
