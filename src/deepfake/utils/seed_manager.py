import random, torch
import numpy as np

class SeedManager:
    """Keep Python, NumPy, and PyTorch RNGs aligned for reproducible pipelines."""

    def __init__(self, seed=42):
        # APPLY THE PROVIDED SEED AS SOON AS THE MANAGER IS CREATED.
        self.set_seed(seed)

    def set_seed(self, seed):
        """Seed all supported RNG backends, toggling PyTorch into deterministic mode when possible."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
