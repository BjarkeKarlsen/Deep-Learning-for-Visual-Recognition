import json
import os


def check_model_exists(model_path):
    """Ensure the expected model checkpoint exists before invoking evaluation."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train/test the model first using `deepfake-cli --train` or `--eval`."
        )