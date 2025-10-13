import json
import os


def check_model_exists(model_path: str) -> bool:
    """Ensure the expected model checkpoint exists before invoking evaluation."""
    
    return os.path.exists(model_path)
        # raise FileNotFoundError(
        #     f"Model file not found at {model_path}. "
        #     "Please train/test the model first using `deepfake-cli --train` or `--eval`."
        # )
