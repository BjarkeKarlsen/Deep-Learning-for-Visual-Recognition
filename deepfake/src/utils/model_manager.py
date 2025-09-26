import json
import os


def check_model_exists(model_path):
    """Fail fast when an evaluation pipeline is invoked before training."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train/test the model first using: python main.py --train or --eval"
        )


def save_training_history(results_dir, history, history_file='training_history.json'):
    """Persist metric history or evaluation payload under the results directory."""

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, history_file), 'w') as f:
        json.dump(history, f, indent=2)
