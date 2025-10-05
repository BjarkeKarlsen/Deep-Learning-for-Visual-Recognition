
import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from deepfake.config import Config

try:
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:  # torch may not be installed in lightweight environments
    torch = None
    _TORCH_AVAILABLE = False

class ConfigLoader:
    """Load configuration by merging defaults with user overrides."""

    DEFAULT_FILENAME = "config.yaml"
    DEFAULT_TEMPLATE = "default.yaml"

    def __init__(self, config_path: Optional[str] = None, *, run_mode: Optional[str] = None):
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.default_config_path = Path(__file__).resolve().parent.parent / "config" / self.DEFAULT_TEMPLATE
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.repo_root = repo_root
        self._default_cfg = None
        self.run_mode = run_mode
        self.cfg = self.load_config()
        self._ensure_output_dirs()

    def load_config(self) -> Config:
        base = OmegaConf.structured(Config)
        defaults_yaml = self._load_yaml(self.default_config_path)
        default_cfg = OmegaConf.merge(base, defaults_yaml)
        self._default_cfg = OmegaConf.to_container(default_cfg, resolve=False)
        cfg = OmegaConf.merge(base, defaults_yaml)

        user_cfg = None
        if self.config_path:
            if not Path(self.config_path).is_file():
                raise FileNotFoundError(f"Config file not found at {self.config_path}")
            user_cfg = self._load_yaml(Path(self.config_path))
            cfg = OmegaConf.merge(cfg, user_cfg)

        user_specified_device = False
        if user_cfg is not None:
            user_dict = OmegaConf.to_container(user_cfg, resolve=False)
            training_cfg = user_dict.get("training") if isinstance(user_dict, dict) else None
            if isinstance(training_cfg, dict) and "device" in training_cfg:
                user_specified_device = True

        if not user_specified_device:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                cfg.training.device = "cuda"
            else:
                cfg.training.device = "cpu"

        # Derive a per-run subdirectory to isolate artifacts, unless disabled.
        # Priority: env RUN_NAME > cfg.paths.run_name > auto timestamped name
        run_name = self._resolve_run_name(cfg)
        if run_name:
            # Append run_name to results/logging dirs and to the parent of the model path
            if cfg.paths.results_dir:
                cfg.paths.results_dir = str(Path(cfg.paths.results_dir) / run_name)
            if cfg.paths.logging_dir:
                cfg.paths.logging_dir = str(Path(cfg.paths.logging_dir) / run_name)
            if cfg.paths.model_path:
                model_path = Path(cfg.paths.model_path)
                cfg.paths.model_path = str(model_path.parent / run_name / model_path.name)
            # Persist the resolved run_name back into cfg
            cfg.paths.run_name = run_name

        cfg_obj: Config = OmegaConf.to_object(cfg)
        self._normalize_paths(cfg_obj)
        return cfg_obj

    def get_config(self) -> Config:
        return self.cfg

    def dump_default_config(self, destination: Optional[Path] = None, *, force: bool = False) -> Path:
        """Write the default configuration template to disk."""
        default_destination = self.repo_root / "configs" / self.DEFAULT_FILENAME
        destination = destination or default_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file at {destination}. Use --force to override."
            )
        OmegaConf.save(OmegaConf.create(self._default_cfg), destination)
        return destination

    def _ensure_output_dirs(self) -> None:
        """Create results/logging directories so subsequent saves do not fail."""
        results_dir = Path(self.cfg.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.paths.logging_dir:
            Path(self.cfg.paths.logging_dir).mkdir(parents=True, exist_ok=True)

        # Ensure the model directory exists too
        model_dir = Path(self.cfg.paths.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of the merged configuration for reproducibility
        self._save_merged_config(results_dir)

    @staticmethod
    def _load_yaml(path: Path):
        """Load a YAML file while surfacing a clear error if it is missing."""
        if not path.exists():
            raise FileNotFoundError(f"Expected configuration file at {path}")
        return OmegaConf.load(path)

    def _normalize_paths(self, cfg: Config) -> None:
        """Resolve relative paths in the config against the repository root."""
        cfg.paths.model_path = self._resolve_repo_path(cfg.paths.model_path)
        cfg.paths.results_dir = self._resolve_repo_path(cfg.paths.results_dir)
        if cfg.paths.logging_dir:
            cfg.paths.logging_dir = self._resolve_repo_path(cfg.paths.logging_dir)

    def _resolve_repo_path(self, path_str: str) -> str:
        """Return an absolute path for repo-relative or user-relative inputs."""
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (self.repo_root / path).resolve()
        return str(path)

    def _resolve_run_name(self, cfg: OmegaConf) -> Optional[str]:
        """Compute a run name used to isolate outputs.

        Precedence:
          1) Environment variable RUN_NAME
          2) cfg.paths.run_name (if provided)
          3) Latest training run when evaluating
          4) Auto-generated timestamp when training or no history exists
        """
        import os

        env_run = os.environ.get("RUN_NAME")
        if env_run:
            return self._sanitize(env_run)

        cfg_run = cfg.paths.get("run_name") if hasattr(cfg, "paths") else None
        if cfg_run:
            return self._sanitize(str(cfg_run))

        # Reuse the latest training run automatically during evaluation.
        if self.run_mode == "eval":
            latest = self._read_latest_run_name(cfg.paths.model_path)
            if latest:
                return latest

        # Auto-generate a minimal name: timestamp only
        # Example: 250104_142355
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        auto = ts
        return self._sanitize(auto)

    @staticmethod
    def _sanitize(name: str) -> str:
        """Make a safe directory name (alnum, dash, underscore, dot)."""
        safe = [c if c.isalnum() or c in ("-", "_", ".") else "-" for c in name.strip()]
        # Collapse consecutive dashes
        out = []
        for ch in safe:
            if not out or not (ch == "-" and out[-1] == "-"):
                out.append(ch)
        return "".join(out).strip("-_")

    def _save_merged_config(self, results_dir: Path) -> None:
        """Write the effective run configuration beside results."""
        if self.run_mode == "eval":
            # Avoid overwriting the training configuration when evaluating.
            return
        try:
            path = results_dir / "run_config.yaml"
            data = asdict(self.cfg)
            OmegaConf.save(OmegaConf.create(data), path)
        except Exception:
            # Do not fail the run if saving the config copy fails
            pass

    @staticmethod
    def _read_latest_run_name(model_path_str: str) -> Optional[str]:
        """Read latest_run.json beside the task directory, if it exists."""
        try:
            model_path = Path(model_path_str)
            task_dir = model_path.parent
            pointer = task_dir / "latest_run.json"
            if pointer.is_file():
                data = json.loads(pointer.read_text())
                run_name = data.get("run_name")
                if isinstance(run_name, str) and run_name.strip():
                    cleaned = ConfigLoader._sanitize(run_name)
                    return cleaned or None
        except Exception:
            return None
        return None


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for dumping or inspecting configuration files."""
    parser = argparse.ArgumentParser(description="Deepfake configuration helper")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a user config file (defaults to repo config.yaml)",
    )
    parser.add_argument(
        "--dump-default",
        action="store_true",
        help="Write the default configuration template to disk",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for --dump-default (defaults to repo config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination when using --dump-default",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    loader = ConfigLoader(config_path=args.config)

    if args.dump_default:
        destination = loader.dump_default_config(
            Path(args.output) if args.output else None,
            force=args.force,
        )
        print(f"Default config written to {destination}")
    else:
        print(OmegaConf.to_yaml(loader.get_config()))


if __name__ == "__main__":
    main()
