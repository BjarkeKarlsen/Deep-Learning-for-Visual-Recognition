from typing import Optional, Literal, Callable
from datasets import Dataset, load_dataset, DownloadMode

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

SplitType = Literal["train", "validation", "test"]


class SIDDatasetManager:
    def __init__(
        self,
        dataset_name: str,
        use_streaming: bool = False,
        cache_dir: Optional[str] = None,
        download_mode: DownloadMode = DownloadMode.REUSE_CACHE_IF_EXISTS,
    ):
        self.dataset_name = dataset_name
        self.use_streaming = use_streaming
        self.cache_dir = cache_dir
        self.download_mode = download_mode

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def get_split(
        self,
        *,
        split_type: SplitType,
        max_samples: Optional[int] = None,
        use_test_or_val_as_test_set: bool = False,
        val_offset: int = 0,
        test_offset: int = 0,
        filter_fn: Optional[Callable] = None,
    ) -> Dataset:
        """
        Load a single split.
        If split_type is 'test' and use_test_or_val_as_test_set=True:
          - derive from 'validation' if val_offset > 0
          - derive from 'train' if val_offset == 0 and test_offset >= 0
        """
        load_n = max_samples * 4 if (filter_fn and max_samples) else max_samples
        
        if split_type == TRAIN:
            ds = self._load_split(TRAIN, max_samples=load_n)
        elif split_type == VALIDATION:
            ds = self._load_split(VALIDATION, max_samples=load_n)
        elif split_type == TEST:
            ds = self._get_test_split(
                max_samples=load_n,
                use_test_or_val_as_test_set=use_test_or_val_as_test_set,
                val_offset=val_offset,
                test_offset=test_offset,
            )
        else:
            raise ValueError(f"Unknown split type: {split_type}")

        # If no filtering, or no limit, return as is
        if filter_fn is None or max_samples is None:
            return ds

        # Apply filtering if provided
        if filter_fn is not None:
            ds = self._filter_dataset(
                dataset=ds,
                split_type=split_type,
                filter_fn=filter_fn
            ).take(max_samples)
            print(f"After filtering, {split_type} dataset size: {len(ds)}")
            
        return ds

    def _load_split(
        self,
        split: str,
        *,
        max_samples: Optional[int] = None,
    ) -> Dataset:
        if not self.use_streaming:
            slice_str = f"{split}[:{max_samples or ''}]"
            print(f"Loading {self.dataset_name} split '{slice_str}' (non-streaming)...")
            ds = load_dataset(
                path=self.dataset_name,
                split=slice_str,
                streaming=False,
                cache_dir=self.cache_dir,
                download_mode=self.download_mode,
            )
        else:
            print(f"Loading {self.dataset_name} split '{split}' (streaming)...")
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

    def _get_test_split(
        self,
        max_samples: Optional[int] = None,
        use_test_or_val_as_test_set: bool = False,
        val_offset: int = 0,
        test_offset: int = 0,
    ) -> Dataset:
        """Derive test from val (if val_offset > 0) or from train (if val_offset == 0)."""
        # Try real test split
        if not use_test_or_val_as_test_set:
            try:
                return self._load_split("test", max_samples=max_samples)
            except Exception:
                raise ValueError("Test split does not exist and use_test_or_val_as_test_set=False.")

        # Case 1: Derive from validation (if val_offset > 0)
        if val_offset > 0:
            if self.use_streaming:
                val_ds = self._load_split(VALIDATION, max_samples=None)
                total = len(val_ds)
                start_idx = max(0, total - val_offset + test_offset)
                derived = val_ds.skip(start_idx)
                if max_samples is not None:
                    derived = derived.take(max_samples)
                return derived
            else:
                full_val = load_dataset(
                    self.dataset_name,
                    split=VALIDATION,
                    streaming=False,
                    cache_dir=self.cache_dir,
                    download_mode=self.download_mode,
                )
                total = len(full_val)
                start = max(0, total - val_offset + test_offset)
                end = min(start + (max_samples or (total - start)), total)
                return full_val.select(range(start, end))

        # Case 2: Derive from train (if val_offset == 0)
        if val_offset == 0 and test_offset >= 0:
            if self.use_streaming:
                train_ds = self._load_split(TRAIN, max_samples=None)
                total = len(train_ds)
                start_idx = max(0, total - test_offset)
                derived = train_ds.skip(start_idx)
                if max_samples is not None:
                    derived = derived.take(max_samples)
                return derived
            else:
                full_train = load_dataset(
                    self.dataset_name,
                    split=TRAIN,
                    streaming=False,
                    cache_dir=self.cache_dir,
                    download_mode=self.download_mode,
                )
                total = len(full_train)
                start = max(0, total - test_offset)
                end = min(start + (max_samples or (total - start)), total)
                return full_train.select(range(start, end))

        raise ValueError("Invalid configuration: cannot derive test split.")


    def _filter_dataset(
        self,
        dataset: Dataset,
        split_type: SplitType,
        filter_fn: Callable,
        target_samples: Optional[int] = None,
    ) -> Dataset:
        """Filter dataset and optionally limit to target number of samples."""
        original_size = len(dataset)
        print(f"Original {split_type} dataset size: {original_size}")
        
        # Apply the filter function
        filtered_dataset = dataset.filter(filter_fn)
        filtered_size = len(filtered_dataset)
        
        print(f"Filtered {split_type} dataset size: {filtered_size}")
        
        # ✅ IMPROVED ERROR HANDLING
        if filtered_size == 0:
            # Provide detailed diagnostic information
            print(f"❌ WARNING: No samples passed the filter for '{split_type}' split")
            
            # Sample a few examples to see what's in the dataset
            if original_size > 0:
                sample = dataset[0]
                print(f"📊 Sample data structure: {sample.keys()}")
                print(f"📊 Sample label: {sample.get('label', 'N/A')}")
                print(f"📊 Sample has mask: {sample.get('mask') is not None}")
                
                # Count labels in the dataset to understand distribution
                if original_size <= 1000:  # Only do this for small datasets
                    label_counts = {}
                    mask_counts = {"has_mask": 0, "no_mask": 0}
                    
                    for i in range(min(100, original_size)):  # Sample first 100
                        example = dataset[i]
                        label = example.get('label', 'unknown')
                        label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if example.get('mask') is not None:
                            mask_counts["has_mask"] += 1
                        else:
                            mask_counts["no_mask"] += 1
                    
                    print(f"📊 Label distribution (first 100 samples): {label_counts}")
                    print(f"📊 Mask distribution (first 100 samples): {mask_counts}")
            
            # ✅ More specific error message
            raise RuntimeError(
                f"No samples passed the filter for '{split_type}' split.\n"
                f"Original size: {original_size}, Filtered size: {filtered_size}\n"
                f"This likely means the {split_type} split contains no tampered images with masks.\n"
                f"Consider:\n"
                f"  1. Using a different split for validation\n"
                f"  2. Not applying the filter to validation split\n"
                f"  3. Checking if your dataset has tampered samples in validation\n"
                f"  4. Increasing filter_multiplier parameter"
            )

        # ✅ NEW: Limit to target number of samples after filtering
        if target_samples is not None and filtered_size > target_samples:
            final_dataset = filtered_dataset.select(range(target_samples))
            print(f"📝 Limited to {target_samples} samples from {filtered_size} filtered samples")
            return final_dataset
        elif target_samples is not None and filtered_size < target_samples:
            print(f"⚠️  Warning: Only found {filtered_size} filtered samples, but {target_samples} were requested")
            print(f"💡 Consider increasing filter_multiplier (currently loading {original_size} samples)")
                
        return filtered_dataset

class DatasetFilters:
    """Common filter functions for different use cases."""
    
    @staticmethod
    def tampered_with_masks(example):
        """Filter for segmentation: only tampered images with valid masks."""
        return example["label"] == 2 and example.get("mask") is not None
    
    @staticmethod
    def classification_only(example):
        """Filter for classification: all images with valid labels."""
        return example.get("label") is not None
    
    @staticmethod
    def real_images_only(example):
        """Filter for real images only."""
        return example["label"] == 0
    
    @staticmethod
    def synthetic_images_only(example):
        """Filter for synthetic images only."""
        return example["label"] == 1