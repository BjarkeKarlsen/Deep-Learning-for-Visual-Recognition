import json
import os
from pathlib import Path
from typing import Optional


def check_model_exists(model_path):
    """Ensure the expected model checkpoint exists before invoking evaluation."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train/test the model first using `deepfake-cli --train` or `--eval`."
        )


def save_training_history(results_dir, history, history_file='training_history.json'):
    """Persist metric history or evaluation payload as JSON under the results directory."""

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, history_file), 'w') as f:
        json.dump(history, f, indent=2)


def _task_dir_from_model_path(model_path: str) -> Path:
    """Return the task-level directory that contains run subfolders.

    Given .../models/<task>/<run_name>/best_model.pth -> returns .../models/<task>
    """
    p = Path(model_path)
    # parent is run folder, parent.parent is task dir
    return p.parent.parent


def _write_latest_run_pointer(model_path: str, run_name: str) -> None:
    """Write a pointer file recording the latest training run for the task."""
    task_dir = _task_dir_from_model_path(model_path)
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "model_path": model_path,
    }
    with open(task_dir / "latest_run.json", "w") as f:
        json.dump(payload, f, indent=2)


def _read_latest_run_pointer(task_dir: Path) -> Optional[str]:
    """Read latest_run.json and return the run_name if available."""
    ptr = task_dir / "latest_run.json"
    if ptr.is_file():
        try:
            data = json.loads(ptr.read_text())
            rn = data.get("run_name")
            if isinstance(rn, str) and rn:
                return rn
        except Exception:
            return None
    return None


def use_latest_run_if_available(logger, cfg) -> bool:
    """If the configured model does not exist, try falling back to the latest run.

    Returns True if cfg.paths.* was updated to point at an existing run.
    """
    model_path = Path(cfg.paths.model_path)
    if model_path.is_file():
        return False

    task_dir = _task_dir_from_model_path(str(model_path))
    latest_run = _read_latest_run_pointer(task_dir)

    if latest_run is None:
        # As a fallback, scan subdirectories for a run containing the model filename
        try:
            candidates = []
            for child in task_dir.iterdir():
                if child.is_dir():
                    candidate = child / model_path.name
                    if candidate.is_file():
                        candidates.append(candidate)
            if candidates:
                # pick the most recent by mtime
                candidate = max(candidates, key=lambda p: p.stat().st_mtime)
                latest_run = candidate.parent.name
        except Exception:
            latest_run = None

    if latest_run:
        # Recompute paths using latest_run
        basename = model_path.name
        new_model = task_dir / latest_run / basename
        if new_model.is_file():
            logger.info(
                f"Model not found at {model_path}. Falling back to latest run: {latest_run}"
            )
            # Update model path
            cfg.paths.model_path = str(new_model)
            # Update run_name
            cfg.paths.run_name = latest_run

            # Results dir: parent is task base; append latest_run
            results_dir = Path(cfg.paths.results_dir)
            cfg.paths.results_dir = str(results_dir.parent / latest_run)

            # Logging dir: if configured, parent is task base; append latest_run
            if getattr(cfg.paths, "logging_dir", None):
                logging_dir = Path(cfg.paths.logging_dir)
                cfg.paths.logging_dir = str(logging_dir.parent / latest_run)
            return True

    return False
