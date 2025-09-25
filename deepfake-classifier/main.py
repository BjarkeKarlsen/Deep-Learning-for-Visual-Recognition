#!/usr/bin/env python3

import os
import argparse

from src.evaluate import evaluate
from src.train import train
from src.utils.seed_manager import SeedManager
from src.utils.logger import SidLogger as SidLogger
from src.utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection CNN')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (defaults to repo config.yaml)')
    args = parser.parse_args()
    
    seed_manager = SeedManager()
    seed_manager.set_seed(42)
    
    loader = ConfigLoader(config_path=args.config)
    cfg = loader.get_config()

    if args.train:
        train(SidLogger('training'), cfg)
    elif args.eval:
        evaluate(SidLogger('evaluation'), cfg)
    else:
        print("Please specify --train or --eval")
        parser.print_help()

if __name__ == "__main__":
    main()
