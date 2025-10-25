from typing import Optional, Literal, Callable
from datasets import Dataset, load_dataset, DownloadMode

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

SplitType = Literal["train", "validation", "test"]


class SIDDatasetManager:
    # CENTRALISES DATASET LOADING, STREAMING HANDLING, AND SPLIT FILTERING.
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
        train_offset: int = 0,
        filter_fn: Optional[Callable] = None,
    ) -> Dataset:
        """
        Load a single split.
        If split_type is 'test' and use_test_or_val_as_test_set=True:
          - derive from 'validation' if val_offset > 0
          - derive from 'train' if val_offset == 0 and test_offset >= 0
        """
        # ESTIMATE HOW MANY SAMPLES TO PRELOAD WHEN FILTERING.
        filter_multiplier = 4
        if max_samples is None:
            load_n = None
        elif filter_fn:
            load_n = max_samples * filter_multiplier
        else:
            load_n = max_samples
        
        # STEP 1: LOAD THE REQUESTED BASE SPLIT.
        if split_type == TRAIN:
            ds = self._load_split(TRAIN, max_samples=load_n)
        elif split_type == VALIDATION:
            ds = self._load_split(VALIDATION, max_samples=load_n)
        elif split_type == TEST:
            return self._get_test_split(
                max_samples=max_samples,
                use_test_or_val_as_test_set=use_test_or_val_as_test_set,
                val_offset=val_offset,
                train_offset=train_offset,
                filter_fn=filter_fn,
            )
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        
        # STEP 2: MATERIALISE STREAMING SPLITS INTO MAP-STYLE DATASETS.
        if self.use_streaming:
            if load_n is None:
                raise ValueError(
                    "max_samples must be specified when use_streaming=True to materialise the dataset."
                )
            ds = ds.take(load_n)
            ds = Dataset.from_list(list(ds))
            print(f"Converted streaming to HF Dataset with {len(ds)} samples")
        
        # STEP 3: RETURN EARLY IF NO FILTERING OR LIMITING IS NEEDED.
        if filter_fn is None and max_samples is None:
            return ds

        # STEP 4: APPLY OPTIONAL FILTERS BEFORE SLICING.
        if filter_fn:
            ds = self._filter_dataset(
                dataset=ds,
                split_type=split_type,
                filter_fn=filter_fn
            )    
            print(f"After filtering, {split_type} dataset size: {len(ds)}")
            

        # STEP 5: SLICE DOWN TO THE REQUESTED SAMPLE COUNT.
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
            print(f"After slicing top {max_samples}, {split_type} size: {len(ds)}")

        return ds

    def _load_split(
        self,
        split: str,
        *,
        max_samples: Optional[int] = None,
    ) -> Dataset:
        # WRAPS load_dataset SO STREAMING AND NON-STREAMING MODES LOOK THE SAME.
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
        return ds

    def _get_test_split(
        self,
        max_samples: Optional[int] = None,
        use_test_or_val_as_test_set: bool = False,
        val_offset: int = 0,
        train_offset: int = 0,
        filter_fn: Optional[Callable] = None,
    ) -> Dataset:
        """
        Build a test split that is disjoint from the validation subset when both
        originate from the same HuggingFace split (validation/train). Filtering
        is applied before slicing so the offset is measured on the filtered view.
        """
        if not use_test_or_val_as_test_set:
            # BASE CASE: USE THE DEDICATED TEST SPLIT IF AVAILABLE.
            if self.use_streaming:
                return self._collect_streaming_slice(
                    split=TEST,
                    start=0,
                    count=max_samples,
                    filter_fn=filter_fn,
                )

            load_n = max_samples
            if max_samples is not None and filter_fn is not None:
                load_n = max_samples * 4
            ds = self._load_split(TEST, max_samples=load_n)
            if filter_fn:
                ds = self._filter_dataset(
                    dataset=ds,
                    split_type=TEST,
                    filter_fn=filter_fn,
                    target_samples=max_samples,
                )
            if max_samples is not None and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            return ds

        # Case 1: Derive from validation (if val_offset > 0)
        if val_offset > 0:
            # USE VALIDATION TAIL AS TEST WHEN NO SEPARATE TEST SPLIT EXISTS.
            if self.use_streaming:
                return self._collect_streaming_slice(
                    split=VALIDATION,
                    start=val_offset,
                    count=max_samples,
                    filter_fn=filter_fn,
                )
            else:
                full_val = load_dataset(
                    self.dataset_name,
                    split=VALIDATION,
                    streaming=False,
                    cache_dir=self.cache_dir,
                    download_mode=self.download_mode,
                )
                if filter_fn:
                    full_val = self._filter_dataset(
                        dataset=full_val,
                        split_type=VALIDATION,
                        filter_fn=filter_fn,
                    )
                total = len(full_val)
                start = min(val_offset, total)
                end = total if max_samples is None else min(start + max_samples, total)
                return full_val.select(range(start, end))

        # Case 2: Derive from train (if val_offset == 0)
        if val_offset == 0 and train_offset >= 0:
            # SHIFT INTO THE TRAINING SPLIT TO CARVE OUT A TEST WINDOW.
            if self.use_streaming:
                return self._collect_streaming_slice(
                    split=TRAIN,
                    start=train_offset,
                    count=max_samples,
                    filter_fn=filter_fn,
                )
            else:
                full_train = load_dataset(
                    self.dataset_name,
                    split=TRAIN,
                    streaming=False,
                    cache_dir=self.cache_dir,
                    download_mode=self.download_mode,
                )
                if filter_fn:
                    full_train = self._filter_dataset(
                        dataset=full_train,
                        split_type=TRAIN,
                        filter_fn=filter_fn,
                    )
                total = len(full_train)
                start = min(train_offset, total)
                end = total if max_samples is None else min(start + max_samples, total)
                return full_train.select(range(start, end))

        raise ValueError("Invalid configuration: cannot derive test split.")

    def _collect_streaming_slice(
        self,
        *,
        split: SplitType,
        start: int,
        count: Optional[int],
        filter_fn: Optional[Callable],
    ) -> Dataset:
        """
        Materialise a streaming split, apply optional filtering, and return a
        contiguous slice [start, start+count) measured over the filtered view.
        """
        # STREAMING HELPERS BUFFER ROWS UNTIL THE REQUESTED WINDOW IS FILLED.
        stream = self._load_split(split, max_samples=None)
        filtered_samples = []
        skipped = 0
        limit = None if count is None else count

        for example in stream:
            if filter_fn and not filter_fn(example):
                continue

            if skipped < start:
                skipped += 1
                continue

            filtered_samples.append(example)
            if limit is not None and len(filtered_samples) >= limit:
                break

        return Dataset.from_list(filtered_samples)


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
        
        # APPLY USER FILTER AND REPORT ANY DATA LOSS.
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

        # LIMIT OR WARN BASED ON THE AVAILABLE FILTERED SAMPLE COUNT.
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
