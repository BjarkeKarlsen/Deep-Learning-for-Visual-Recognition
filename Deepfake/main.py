#!/usr/bin/env python3
import argparse

from src.evaluate import evaluate
from src.train import train

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection CNN')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        evaluate()
    else:
        print("Please specify --train or --eval")
        parser.print_help()

if __name__ == "__main__":
    main()