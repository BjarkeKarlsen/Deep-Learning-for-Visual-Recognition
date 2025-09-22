import os
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset as HFDataset,
    Dataset,
)
from typing import Dict, Tuple, Optional


class SIDDatasetManager:
    """
    Manages loading, splitting, optional streaming, and on-disk caching
    for the Hugging Face SID dataset.
    """
    def __init__(
        self,
        dataset_name: str = "saberzl/SID_Set",
        cache_dir: str = "data",
        use_disk_cache: bool = True,
        use_streaming: bool = False,
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.use_disk_cache = use_disk_cache
        self.use_streaming = use_streaming
        self._cached_splits: Dict[str, HFDataset] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, split: str) -> str:
        return os.path.join(self.cache_dir, f"SID_{split}")

    def load_dataset(self) -> Dict[str, HFDataset]:
        """
        Load (streaming or regular) and cache full HF splits.
        """
        if not self._cached_splits:
            print(f"Loading full dataset {self.dataset_name} from Hugging Face...")
            full = load_dataset(
                self.dataset_name,
                streaming=self.use_streaming
            )
            for split, ds in full.items():
                self._cached_splits[split] = ds
        return self._cached_splits

    def _get_base_split(
        self,
        split: str
    ) -> HFDataset:
        """
        Return base HF split, streaming or in-memory.
        """
        splits = self.load_dataset()
        return splits[split]

    def _slice_split(
        self,
        ds: HFDataset,
        max_samples: Optional[int],
        start: int = 0
    ) -> HFDataset:
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
    ) -> HFDataset:
        """
        Load (or derive) a split, optionally cache it, respecting streaming flag.
        """
        path = self._cache_path(split)
        # 1) Load from disk if cached
        if self.use_disk_cache and os.path.isdir(path) and os.listdir(path):
            print(f"Loading {split} split from disk cache: {path}")
            return load_from_disk(path)

        # 2) Build the split
        if derive_from:
            base = self._get_base_split(derive_from)
            ds = self._slice_split(base, max_samples, start)
        else:
            base = self._get_base_split(split)
            ds = self._slice_split(base, max_samples)

        # 3) Only cache non-empty splits
        try:
            length = len(ds)
        except TypeError:
            # For streaming ds, convert to list temporarily to get length
            tmp = list(ds.take(max_samples or 1))
            length = len(tmp)
        if self.use_disk_cache and length > 0:
            print(f"Caching {split} split to disk at: {path}")
            Dataset.from_dict(ds[:]).save_to_disk(path)
        else:
            if length == 0:
                print(f"Skipping disk cache for empty {split} split")

        return ds

    def get_splits(
        self,
        train_max: Optional[int] = None,
        val_max: Optional[int] = None,
        test_max: Optional[int] = None,
        val_offset: int = 30_000,
        test_offset: int = 0,
        use_official_test: bool = False
    ) -> Tuple[HFDataset, HFDataset, HFDataset]:
        """
        Return (train, validation, test) datasets, streaming or not,
        with optional on-disk caching and official-test fallback.
        """
        splits = self.load_dataset()

        # TRAIN
        train_ds = self._load_or_cache_split(
            "train", max_samples=train_max
        )

        # VALIDATION
        if "validation" in splits:
            val_ds = self._load_or_cache_split(
                "validation", max_samples=val_max
            )
        else:
            val_ds = self._load_or_cache_split(
                split="validation",
                max_samples=val_max,
                derive_from="train",
                start=len(train_ds) - val_offset
            )

        # TEST
        if use_official_test and "test" in splits:
            test_ds = self._load_or_cache_split(
                "test", max_samples=test_max
            )
        else:
            base = "validation" if "validation" in splits else "train"
            test_offset = val_max if val_max is not None else 0
            test_ds = self._load_or_cache_split(
                split="test",
                max_samples=test_max,
                derive_from=base,
                start=test_offset
            )

        return train_ds, val_ds, test_ds
