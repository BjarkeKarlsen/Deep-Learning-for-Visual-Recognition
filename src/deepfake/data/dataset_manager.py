from typing import Optional, Tuple
from datasets import load_dataset, DownloadMode, IterableDataset, Dataset

class SIDDatasetManager:
    def __init__(
        self,
        dataset_name: str,
        use_streaming: bool = False,
        cache_dir: Optional[str] = None,
        download_mode: DownloadMode = DownloadMode.REUSE_DATASET_IF_EXISTS,
    ):
        self.dataset_name = dataset_name
        self.use_streaming = use_streaming
        self.cache_dir = cache_dir
        self.download_mode = download_mode

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def _load_split(
        self,
        split: str,
        *,
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """
        Load a single split, optionally capped via slice notation (map‐style)
        or via .take() (streaming).
        """
        if not self.use_streaming:
            # map‐style supports slice dicts
            slice_str = f"{split}[:{max_samples or ''}]"
            print( f"Loading {self.dataset_name} split '{slice_str}' (non-streaming)..." )
            ds = load_dataset(
                path=self.dataset_name,
                split=slice_str,
                streaming=False,
                cache_dir=self.cache_dir,
                download_mode=self.download_mode,
            )
        else:
            # streaming must load one split at a time
            ds = load_dataset(
                path=self.dataset_name,
                split=split,
                streaming=True,
                cache_dir=self.cache_dir,
                download_mode=self.download_mode,
            )
            if max_samples is not None:
                ds = ds.take(max_samples)
        return ds

    def get_splits(
        self,
        *,
        train_max: Optional[int] = None,
        val_max: Optional[int] = None,
        test_max: Optional[int] = None,
        derive_test: bool = False,
        val_offset: int = 30000,
        test_offset: int = 0,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Returns (train, val, test), each split loaded with at most N samples.
        If derive_test=True and no 'test' split exists, carve it from train tail.
        """
        # Load or cap each split independently
        train_ds = self._load_split("train", max_samples=train_max)
        val_ds   = self._load_split("validation", max_samples=val_max)
        try:
            test_ds = self._load_split("test", max_samples=test_max)
        except ValueError:
            test_ds = None

        if test_ds is not None or not derive_test:
            return train_ds, val_ds, test_ds or Dataset.from_list([])

        # derive test from end of train (after val_offset)
        if self.use_streaming:
            # streaming: skip first (train_max or all minus val_offset) + test_offset, then take test_max
            skip_n = (train_max or 0) - val_offset + test_offset
            test_ds = train_ds.skip(skip_n)
            if test_max is not None:
                test_ds = test_ds.take(test_max)
        else:
            full_train = load_dataset(self.dataset_name, split="train", streaming=False)
            total = len(full_train)
            start = max(0, total - val_offset + test_offset)
            end   = start + (test_max or (total - start))
            test_ds = full_train.select(range(start, end))

        return train_ds, val_ds, test_ds
