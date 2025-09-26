#!/usr/bin/env python3

import os
import argparse

import torch.multiprocessing as mp

from src.classification import evaluate as classify_evaluate
from src.classification import train as classify_train
from src.segmentation import evaluate as segment_evaluate
from src.segmentation import train as segment_train
from src.utils.seed_manager import SeedManager
from src.utils.logger import SidLogger as SidLogger
from src.utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection CNN')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--task', type=str, required=True,
                        choices=['classification', 'segmentation'],
                        help='Which pipeline to run')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML (override for defaults)')
    args = parser.parse_args()
    loader = ConfigLoader(config_path=args.config)
    cfg = loader.get_config()

    seed_value = getattr(cfg.training, "seed", None)
    SeedManager(seed_value if seed_value is not None else 42)

    log_dir = cfg.paths.logging_dir or "logs"

    if args.train or args.eval:
        task = args.task
        if task == 'classification':
            if args.train:
                classify_train(SidLogger('training', log_dir=log_dir), cfg)
            if args.eval:
                classify_evaluate(SidLogger('evaluation', log_dir=log_dir), cfg)
        else:  # segmentation
            if args.train:
                segment_train(SidLogger('segmentation-train', log_dir=log_dir), cfg)
            if args.eval:
                segment_evaluate(SidLogger('segmentation-eval', log_dir=log_dir), cfg)
    else:
        print("Please specify --train or --eval")
        parser.print_help()

if __name__ == "__main__":
    start_method = mp.get_start_method(allow_none=True)
    if start_method != "spawn":
        mp.set_start_method("spawn", force=True)  # safe default across platforms when using PyTorch + multiprocessing
    main()
