import os
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
)
from typing import Dict, Tuple, Optional


class SIDDatasetManager:
    """
    Manages loading, splitting, optional streaming, and caching
    for the Hugging Face SID dataset, always using HF cache locations.
    """
    def __init__(
        self,
        dataset_name: str,
        use_disk_cache: bool = True,
        use_streaming: bool = False,
    ):
        self.dataset_name = dataset_name
        self.use_disk_cache = use_disk_cache
        self.use_streaming = use_streaming
        self._cached_splits: Dict[str, Dataset] = {}

    def _get_hf_cache_dir(self) -> str:
        """Get the HuggingFace cache directory being used."""
        # Check environment variables in order of precedence
        hf_datasets_cache = os.environ.get('HF_DATASETS_CACHE')
        if hf_datasets_cache:
            return hf_datasets_cache
            
        hf_home = os.environ.get('HF_HOME')
        if hf_home:
            return os.path.join(hf_home, 'datasets')
            
        # Default HF location
        return os.path.expanduser('~/.cache/huggingface/datasets')

    def _cache_path(self, split: str) -> str:
        """Generate cache path for custom splits within HF cache structure."""
        hf_cache = self._get_hf_cache_dir()
        return os.path.join(hf_cache, "custom_splits", "SID", f"{split}")

    def load_dataset(self) -> Dict[str, Dataset]:
        """
        Load (streaming or regular) dataset using HF's caching system.
        """
        if not self._cached_splits:
            print(f"Loading dataset {self.dataset_name} from Hugging Face...")
            
            # Always use HuggingFace's built-in caching (cache_dir omitted)
            full = load_dataset(
                self.dataset_name,
                streaming=self.use_streaming
                # cache_dir is omitted, so HF uses default location
            )
                
            for split, ds in full.items():
                self._cached_splits[split] = ds
                
        return self._cached_splits

    def _get_base_split(self, split: str) -> Dataset:
        """Return base HF split, streaming or in-memory."""
        splits = self.load_dataset()
        return splits[split]

    def _slice_split(
        self,
        ds: Dataset,
        max_samples: Optional[int],
        start: int = 0
    ) -> Dataset:
        """
        For streaming: use .take(max_samples) then Dataset.from_list.
        For regular: use .select(range(start, start+max_samples)).
        """
        if self.use_streaming:
            if max_samples is None:
                return ds
            items = list(ds.take(max_samples) if start == 0
                         else ds.skip(start).take(max_samples))
            return Dataset.from_list(items)
        else:
            if max_samples is None:
                return ds
            end = start + max_samples
            return ds.select(range(start, min(end, len(ds))))

    def _load_or_cache_split(
        self,
        split: str,
        max_samples: Optional[int] = None,
        derive_from: Optional[str] = None,
        start: int = 0
    ) -> Dataset:
        """
        Load (or derive) a split, optionally cache it in HF cache structure.
        """
        path = self._cache_path(split)
        
        # 1) Load from disk if cached (only for custom splits)
        if (self.use_disk_cache and 
            derive_from is not None and  # Only custom splits are cached
            os.path.isdir(path) and os.listdir(path)):
            print(f"Loading {split} split from HF cache: {path}")
            return load_from_disk(path)

        # 2) Build the split
        if derive_from:
            base = self._get_base_split(derive_from)
            ds = self._slice_split(base, max_samples, start)
        else:
            base = self._get_base_split(split)
            ds = self._slice_split(base, max_samples)

        # 3) Cache custom splits in HF cache structure
        if derive_from and self.use_disk_cache:
            try:
                length = len(ds)
            except TypeError:
                # For streaming ds, convert to list temporarily to get length
                tmp = list(ds.take(max_samples or 1))
                length = len(tmp)
                
            if length > 0:
                # Ensure cache directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                print(f"Caching custom {split} split to HF cache: {path}")
                ds.save_to_disk(path)
            else:
                print(f"Skipping cache for empty {split} split")

        return ds

    def get_splits(
        self,
        train_max: Optional[int] = None,
        val_max: Optional[int] = None,
        test_max: Optional[int] = None,
        val_offset: int = 30_000,
        test_offset: int = 0,
        use_official_test: bool = False
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Return (train, validation, test) datasets, all cached in HF locations.
        """
        splits = self.load_dataset()

        # TRAIN - use HF caching for original split
        if train_max is None:
            train_ds = self._get_base_split("train")
        else:
            train_ds = self._load_or_cache_split(
                "train_subset", max_samples=train_max, derive_from="train"
            )

        # VALIDATION
        if "validation" in splits and val_max is None:
            val_ds = self._get_base_split("validation")
        elif "validation" in splits:
            val_ds = self._load_or_cache_split(
                "validation_subset", max_samples=val_max, derive_from="validation"
            )
        else:
            val_ds = self._load_or_cache_split(
                split="validation_custom",
                max_samples=val_max,
                derive_from="train",
                start=len(self._get_base_split("train")) - val_offset
            )

        # TEST
        if use_official_test and "test" in splits and test_max is None:
            test_ds = self._get_base_split("test")
        elif use_official_test and "test" in splits:
            test_ds = self._load_or_cache_split(
                "test_subset", max_samples=test_max, derive_from="test"
            )
        else:
            base = "validation" if "validation" in splits else "train"
            test_offset_actual = val_max if val_max is not None else test_offset
            test_ds = self._load_or_cache_split(
                split="test_custom",
                max_samples=test_max,
                derive_from=base,
                start=test_offset_actual
            )

        return train_ds, val_ds, test_ds

    def get_cache_info(self) -> Dict[str, str]:
        """Get information about cache directories being used."""
        hf_cache = self._get_hf_cache_dir()
        custom_splits_cache = os.path.join(hf_cache, "custom_splits", "SID")
        
        info = {
            "hf_datasets_cache": hf_cache,
            "custom_splits_cache": custom_splits_cache,
        }
        
        # Add environment variables info
        for env_var in ['HF_HOME', 'HF_DATASETS_CACHE']:
            info[env_var] = os.environ.get(env_var, "Not set")
            
        return info

    def clear_custom_cache(self):
        """Clear only the custom split caches, preserve original HF dataset cache."""
        import shutil
        
        custom_cache_root = os.path.join(self._get_hf_cache_dir(), "custom_splits", "SID")
            
        if os.path.exists(custom_cache_root):
            shutil.rmtree(custom_cache_root)
            print(f"Cleared custom splits cache: {custom_cache_root}")
        else:
            print("No custom splits cache to clear")