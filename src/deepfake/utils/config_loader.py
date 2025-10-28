
import argparse
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

    def __init__(self, config_path: Optional[str] = None):
        # INITIALISE PATHS TO DEFAULT RESOURCES AND USER OVERRIDES.
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.default_config_path = Path(__file__).resolve().parent.parent / "config" / self.DEFAULT_TEMPLATE
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.repo_root = repo_root
        self._default_cfg = None
        self.cfg = self.load_config()

    def load_config(self) -> Config:
        # MERGE STRUCTURED DEFAULTS WITH OPTIONAL USER SUPPLIED YAML.
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
            # AUTO-SELECT DEVICE WHEN THE USER LEAVES IT UNSPECIFIED.
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                cfg.training.device = "cuda"
            else:
                cfg.training.device = "cpu"

        cfg_obj: Config = OmegaConf.to_object(cfg)
        return cfg_obj

    def get_config(self) -> Config:
        return self.cfg

    def dump_default_config(self, destination: Optional[Path] = None, *, force: bool = False) -> Path:
        """Write the default configuration template to disk."""
        # ALLOWS USERS TO INSPECT OR CUSTOMISE THE BASE CONFIG TEMPLATE.
        default_destination = self.repo_root / "configs" / self.DEFAULT_FILENAME
        destination = destination or default_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file at {destination}. Use --force to override."
            )
        OmegaConf.save(OmegaConf.create(self._default_cfg), destination)
        return destination

    @staticmethod
    def _load_yaml(path: Path):
        """Load a YAML file while surfacing a clear error if it is missing."""
        if not path.exists():
            raise FileNotFoundError(f"Expected configuration file at {path}")
        return OmegaConf.load(path)

    # def _normalize_paths(self, cfg: Config) -> None:
    #     """Resolve relative paths in the config against the repository root."""
    #     cfg.paths.model_filename = self._resolve_repo_path(cfg.paths.model_filename)
        #cfg.paths.output_dir = self._resolve_repo_path(cfg.paths.output_dir)
        #if cfg.paths.log_dir:
        #    cfg.paths.log_dir = self._resolve_repo_path(cfg.paths.log_dir)

    def _resolve_repo_path(self, path_str: str) -> str:
        """Return an absolute path for repo-relative or user-relative inputs."""
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (self.repo_root / path).resolve()
        return str(path)


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
