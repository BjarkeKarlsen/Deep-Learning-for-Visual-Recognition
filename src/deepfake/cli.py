#!/usr/bin/env python3
"""Command-line entry point for the Deepfake detection pipelines."""

import argparse
from deepfake.visualization.segmentation_plots import SegmentationPlots
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
        '--env',
        type=str,
        choices=["dev", "test", ],
        help='Environment to use for paths and logging',
        default="dev"
    )
    
    parser.add_argument(
        '--plot',
        type=str,
        choices=["classification", "segmentation"],
        help='Create plots from training history JSON file.'
    )
        
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     help="Path to a YAML configuration override",
    # )
    args = parser.parse_args()
    
    if args.env is None:
        print("Please specify an environment using --env [dev|test]")
        parser.print_help()
        return
    
    config_file = "configs/"
    if args.task:
        if args.env == "dev":
            config_file += args.env + "-" + args.task
        elif args.env == "test":
            config_file += args.task 
    else:
        if args.plot:
            if args.env == "dev":
                config_file += args.env + "-" + args.plot
            elif args.env == "test":
                config_file += args.plot
    config_file += ".yaml"

    loader = ConfigLoader(config_path=config_file)
    cfg = loader.get_config()

    seed_value = getattr(cfg.training, "seed", None) or 42
    SeedManager(seed_value)

    if args.task == cfg.Task.CLASSIFICATION:
        if args.train:
            classify_train(SidLogger("classification-train", log_dir=cfg.paths.log_dir), cfg)
        if args.eval:
            classify_evaluate(SidLogger("classification-eval", log_dir=cfg.paths.log_dir), cfg)
    elif args.task == cfg.Task.SEGMENTATION:
        if args.train:
            segment_train(SidLogger("segmentation-train", log_dir=cfg.paths.log_dir), cfg)
        if args.eval:
            segment_evaluate(SidLogger("segmentation-eval", log_dir=cfg.paths.log_dir), cfg)
    elif args.plot:
        if args.plot == cfg.Task.CLASSIFICATION:
            
            if args.eval:
                class_plots = ClassificationPlots(output_directory=cfg.paths.output_dir, eval_history_path=cfg.paths.eval_metrics_path)
                print("Plotting classification metrics eval...")
                class_plots.plot_confusion_matrix()
                class_plots.plot_classification_report()
                
            elif args.train:
                class_plots = ClassificationPlots(training_history_path=cfg.paths.history_path, output_directory=cfg.paths.output_dir)
                print("Plotting classification metrics train...")
                class_plots.plot_training_history()
                class_plots.plot_learning_rate_schedule()
        elif args.plot == cfg.Task.SEGMENTATION:
            
            if args.eval:
                segmentation_plots = SegmentationPlots(eval_history_path=cfg.paths.eval_metrics_path, output_directory=cfg.paths.output_dir)
                print("Plotting segmentation metrics eval...")
                print("Not implemented yet.")

            if args.train:
                segmentation_plots = SegmentationPlots(training_history_path=cfg.paths.history_path, output_directory=cfg.paths.output_dir)
                print("Plotting segmentation metrics train...")
                segmentation_plots.plot_training_history()
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
