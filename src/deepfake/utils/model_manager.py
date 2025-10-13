import json
import os


def check_model_exists(model_path: str) -> bool:
    """Ensure the expected model checkpoint exists before invoking evaluation."""
    
    return os.path.exists(model_path)
