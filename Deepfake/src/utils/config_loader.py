
import os
import torch

from omegaconf import OmegaConf
from config import Config

class ConfigLoader:
    """Load and merge config from YAML file with defaults from dataclass."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.cfg = self.load_config()
        
        # ensure results & logging dirs exist
        

    def load_config(self) -> Config:
        base = OmegaConf.structured(Config)
        if os.path.exists(self.config_path):
            overrides = OmegaConf.load(self.config_path)
            cfg = OmegaConf.merge(base, overrides)
        else:
            cfg = base
        # auto detect device
        cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cfg
    
    def get_config(self) -> Config:
        return self.cfg
    
    def set_up_dir(self):
        os.makedirs(self.cfg.paths.results_dir, exist_ok=True)
        if self.cfg.paths.logging_dir:
            os.makedirs(self.cfg.paths.logging_dir, exist_ok=True)

        # IDK IF KEEP
    def get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
