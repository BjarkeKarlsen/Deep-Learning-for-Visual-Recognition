import random, os
import torch

import numpy as np

from omegaconf import OmegaConf
from config import Config

def load_config(config_path: str = "config.yaml") -> Config:
    base = OmegaConf.structured(Config)
    if os.path.exists(config_path):
        overrides = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base, overrides)
    else:
        cfg = base
    # auto detect device
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def print_gpu_info(device):
    if device == 'cuda':
        print(f"USING GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"USING CPU (No GPU detected)")
        
def print_run_info(info, device):
    print("="*60)
    print(info)
    print("="*60)
    print_gpu_info(device)
    print("="*60)