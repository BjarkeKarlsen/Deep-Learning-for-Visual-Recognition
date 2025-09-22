from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class SIDConfig:
    """
    Global constants and configuration for the SID dataset.
    Use this class throughout your codebase for consistent parameters.
    """
    # Hugging Face dataset identifier
    DATASET_NAME: str = "saberzl/SID_Set"

    # Published split sizes
    TRAIN_SIZE: int      = 210_000
    VAL_SIZE: int        = 30_000
    TEST_SIZE: int       = 60_000

    # Fallback offsets for deriving splits
    VAL_OFFSET: int      = 30_000  # last VAL_OFFSET of train → validation
    TEST_OFFSET: int     = 0       # first TEST_OFFSET of val/train → test

    # Local cache directory
    CACHE_DIR: str       = "data"

    # Image transforms
    IMAGE_SIZE: Tuple[int,int]     = (256, 256)
    NORMALIZE_MEAN: Tuple[float,...] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float,...]  = (0.229, 0.224, 0.225)

    # Disk‐cache toggles
    USE_DISK_CACHE: bool = True
    USE_OFFICIAL_TEST: bool = False

# Usage:
# from config import SIDConfig
# cfg = SIDConfig()
# manager = SIDDatasetManager(
#     dataset_name=cfg.DATASET_NAME,
#     cache_dir=cfg.CACHE_DIR,
#     use_disk_cache=cfg.USE_DISK_CACHE
# )
# train_ds, val_ds, test_ds = manager.get_splits(
#     train_max=cfg.TRAIN_SIZE,
#     val_max=cfg.VAL_SIZE,
#     test_max=cfg.TEST_SIZE,
#     val_offset=cfg.VAL_OFFSET,
#     test_offset=cfg.TEST_OFFSET,
#     use_official_test=cfg.USE_OFFICIAL_TEST
# )
