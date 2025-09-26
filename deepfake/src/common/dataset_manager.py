import os
from itertools import islice
from typing import Dict, List, Optional, Tuple

from datasets import (
    Dataset,
    IterableDataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)


def _has_mask_with_label(example, *, target_label: int) -> bool:
    mask = example.get("mask")
    return mask is not None and example.get("label") == target_label


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

    def _ensure_indexable_split(self, split: str) -> Dataset:
        """Materialise streaming splits into map-style datasets when needed."""
        ds = self._get_base_split(split)
        if isinstance(ds, IterableDataset):
            # Materialise entire split so downstream code can rely on len/index.
            materialised = self._slice_split(ds, max_samples=None)
            self._cached_splits[split] = materialised
            return materialised
        return ds

    @staticmethod
    def _compute_holdout_ranges(
        *,
        total_length: int,
        holdout_start: int,
        val_max: Optional[int],
        derive_test: bool,
        test_max: Optional[int],
        test_offset: int,
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Compute disjoint ranges for validation and test subsets."""

        holdout_start = max(0, min(holdout_start, total_length))
        available = max(0, total_length - holdout_start)
        if available == 0:
            empty_range = (holdout_start, holdout_start)
            return empty_range, None

        offset_reserved = min(available, max(0, test_offset) if derive_test else 0)
        if derive_test and test_max is not None:
            potential_after_offset = max(0, available - offset_reserved)
            reserve_for_test_samples = min(test_max, potential_after_offset)
        else:
            reserve_for_test_samples = 0

        reserve_for_test = min(available, offset_reserved + reserve_for_test_samples)

        max_val_allowed = max(0, available - reserve_for_test)
        if val_max is None:
            val_len = max_val_allowed
        else:
            val_len = min(val_max, max_val_allowed)

        available_after_val = max(0, available - val_len)
        if derive_test and available_after_val > test_offset:
            max_test_len = available_after_val - test_offset
            if test_max is None:
                test_len = max_test_len
            else:
                test_len = min(test_max, max_test_len)
        else:
            test_len = 0

        val_start = holdout_start
        val_end = val_start + val_len
        test_start = min(total_length, val_end + max(0, test_offset))
        test_end = min(total_length, test_start + test_len)

        val_range = (val_start, val_end)
        test_range = (test_start, test_end) if test_len > 0 else None
        return val_range, test_range

    @staticmethod
    def _build_train_excluding(
        ds: Dataset,
        exclude_ranges: List[Tuple[int, int]],
    ) -> Dataset:
        """Return dataset with the provided ranges (inclusive/exclusive) removed."""

        if not exclude_ranges:
            return ds

        length = len(ds)
        normalised = []
        for start, end in exclude_ranges:
            if end <= start:
                continue
            normalised.append((max(0, start), min(length, end)))
        if not normalised:
            return ds

        normalised.sort()
        keep_ranges: List[Tuple[int, int]] = []
        cursor = 0
        for start, end in normalised:
            if cursor < start:
                keep_ranges.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < length:
            keep_ranges.append((cursor, length))

        if not keep_ranges:
            return ds.select([])

        segments = [ds.select(range(start, end)) for start, end in keep_ranges]
        if len(segments) == 1:
            return segments[0]
        return concatenate_datasets(segments)

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

    def _cache_path(self, split: str, *, identifier: Optional[str] = None) -> str:
        """Generate cache path scoped by dataset + split parameters."""
        hf_cache = self._get_hf_cache_dir()
        dataset_slug = self.dataset_name.replace("/", "__")
        parts = [split]
        if identifier:
            parts.append(identifier)
        cache_dir = os.path.join(hf_cache, "custom_splits", "SID", dataset_slug, "__".join(parts))
        return cache_dir

    def load_dataset(self) -> Dict[str, Dataset]:
        """
        Load (streaming or regular) dataset using HF's caching system.
        """
        if not self._cached_splits:
            print(f"Loading dataset {self.dataset_name} from Hugging Face...")

            full = load_dataset(
                self.dataset_name,
                streaming=self.use_streaming,
            )

            for split, ds in full.items():
                if isinstance(ds, IterableDataset) and not self.use_streaming:
                    # Hugging Face sometimes returns iterables even for local
                    # caches; materialise once so PyTorch-style random access
                    # (len, indexing, shuffling) remains available downstream.
                    records = list(ds)
                    ds = Dataset.from_list(records)
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
        if isinstance(ds, IterableDataset):
            iterator = ds
            if start:
                if hasattr(iterator, "skip"):
                    iterator = iterator.skip(start)
                else:
                    iterator = islice(iterator, start, None)

            if max_samples is None:
                items = list(iterator)
            else:
                if hasattr(iterator, "take"):
                    iterator = iterator.take(max_samples)
                    items = list(iterator)
                else:
                    items = list(islice(iterator, max_samples))

            return Dataset.from_list(items)
        else:
            length = len(ds)
            start = max(0, min(start, length))
            if max_samples is None:
                if start == 0:
                    return ds
                end = length
            else:
                end = min(start + max_samples, length)

            if start == 0 and end == length:
                return ds

            return ds.select(range(start, end))

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
        id_components = [derive_from or split]
        max_str = str(max_samples) if max_samples is not None else "all"
        id_components.append(f"max{max_str}")
        if start:
            id_components.append(f"start{start}")
        if self.use_streaming:
            id_components.append("stream")
        identifier = "__".join(id_components)

        path = self._cache_path(split, identifier=identifier)
        
        # 1) Load from disk if cached (only for custom splits)
        if (
            self.use_disk_cache
            and derive_from is not None  # Only custom splits are cached
            and os.path.isdir(path)
            and os.listdir(path)
        ):
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
        has_validation = "validation" in splits
        use_official_test_split = use_official_test and "test" in splits

        # Prepare validation (and optional derived test) from its source split.
        if has_validation:
            needs_holdout_math = (val_max is None) or (not use_official_test_split)
            if needs_holdout_math:
                val_source = self._ensure_indexable_split("validation")
                val_length = len(val_source)
                val_range, validation_test_range = self._compute_holdout_ranges(
                    total_length=val_length,
                    holdout_start=0,
                    val_max=val_max,
                    derive_test=not use_official_test_split,
                    test_max=test_max if not use_official_test_split else None,
                    test_offset=test_offset,
                )

                val_start, val_end = val_range
                val_len = max(0, val_end - val_start)

                if val_start == 0 and val_len == val_length:
                    val_ds = val_source
                else:
                    val_ds = self._load_or_cache_split(
                        "validation_subset",
                        max_samples=val_len,
                        derive_from="validation",
                        start=val_start,
                    )

                if not use_official_test_split:
                    if validation_test_range is not None:
                        test_start, test_end = validation_test_range
                        test_len = max(0, test_end - test_start)
                        test_ds = self._load_or_cache_split(
                            "test_custom",
                            max_samples=test_len,
                            derive_from="validation",
                            start=test_start,
                        )
                    else:
                        test_ds = val_source.select([])
                else:
                    test_ds = None  # Filled later from official split.
            else:
                # Simple subset of validation; no need to materialise entire split.
                val_ds = self._load_or_cache_split(
                    "validation_subset",
                    max_samples=val_max,
                    derive_from="validation",
                    start=0,
                )
                test_ds = None

            # Training split: derived directly from HF train split.
            if train_max is None:
                train_ds = self._ensure_indexable_split("train")
            else:
                train_ds = self._load_or_cache_split(
                    "train_subset", max_samples=train_max, derive_from="train"
                )

        else:
            # Derive validation/test from the training split tail.
            train_indexable = self._ensure_indexable_split("train")
            train_length = len(train_indexable)
            holdout_start = max(0, train_length - val_offset)
            val_range, test_range = self._compute_holdout_ranges(
                total_length=train_length,
                holdout_start=holdout_start,
                val_max=val_max,
                derive_test=not use_official_test_split,
                test_max=test_max if not use_official_test_split else None,
                test_offset=test_offset,
            )

            val_start, val_end = val_range
            val_len = max(0, val_end - val_start)
            val_ds = self._load_or_cache_split(
                "validation_custom",
                max_samples=val_len,
                derive_from="train",
                start=val_start,
            )

            if not use_official_test_split:
                if test_range is not None:
                    test_start, test_end = test_range
                    test_len = max(0, test_end - test_start)
                    test_ds = self._load_or_cache_split(
                        "test_custom",
                        max_samples=test_len,
                        derive_from="train",
                        start=test_start,
                    )
                else:
                    test_ds = train_indexable.select([])
            else:
                test_ds = None

            exclude_ranges: List[Tuple[int, int]] = [val_range]
            if test_range is not None:
                exclude_ranges.append(test_range)

            train_without_holdout = self._build_train_excluding(
                train_indexable, exclude_ranges
            )

            if train_max is None:
                train_ds = train_without_holdout
            else:
                train_ds = self._slice_split(train_without_holdout, max_samples=train_max)

        # If requested, use the official test split (optionally sub-sampled).
        if use_official_test_split:
            if test_max is None:
                test_ds = self._ensure_indexable_split("test")
            else:
                test_ds = self._load_or_cache_split(
                    "test_subset", max_samples=test_max, derive_from="test"
                )

        return train_ds, val_ds, test_ds

    def _filter_segmentation_split(
        self,
        ds: Dataset,
        *,
        cache_key: str,
        tampered_label: int,
        max_samples: Optional[int],
    ) -> Dataset:
        if ds is None:
            return ds

        id_components = [f"label{tampered_label}"]
        if max_samples is not None:
            id_components.append(f"max{max_samples}")
        if self.use_streaming:
            id_components.append("stream")
        cache_path = self._cache_path(cache_key, identifier="__".join(id_components))
        if (
            self.use_disk_cache
            and os.path.isdir(cache_path)
            and os.listdir(cache_path)
        ):
            print(f"Loading {cache_key} split from HF cache: {cache_path}")
            return load_from_disk(cache_path)

        if isinstance(ds, IterableDataset):
            iterator = (
                example
                for example in ds
                if _has_mask_with_label(example, target_label=tampered_label)
            )
            if max_samples is not None:
                iterator = islice(iterator, max_samples)
            records = list(iterator)
            filtered = Dataset.from_list(records)
        else:
            filtered = ds.filter(
                _has_mask_with_label,
                fn_kwargs={"target_label": tampered_label},
            )
            if max_samples is not None and len(filtered) > max_samples:
                filtered = filtered.select(range(max_samples))

        if self.use_disk_cache:
            if len(filtered) > 0:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                print(f"Caching custom {cache_key} split to HF cache: {cache_path}")
                filtered.save_to_disk(cache_path)
            else:
                print(f"Skipping cache for empty {cache_key} split")

        return filtered

    def get_segmentation_splits(
        self,
        *,
        tampered_label: int = 2,
        train_max: Optional[int] = None,
        val_max: Optional[int] = None,
        test_max: Optional[int] = None,
        val_offset: int = 30_000,
        test_offset: int = 0,
        use_official_test: bool = False,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Return splits filtered to samples that have masks for tampering."""

        base_train, base_val, base_test = self.get_splits(
            train_max=None,
            val_max=None,
            test_max=None,
            val_offset=val_offset,
            test_offset=test_offset,
            use_official_test=use_official_test,
        )

        # Each segmentation subset keeps only samples where the selected label
        # includes a tamper mask. This avoids wasting GPU time on negatives that
        # cannot contribute to pixel-level supervision.
        train_ds = self._filter_segmentation_split(
            base_train,
            cache_key="train_tampered",
            tampered_label=tampered_label,
            max_samples=train_max,
        )

        val_ds = self._filter_segmentation_split(
            base_val,
            cache_key="validation_tampered",
            tampered_label=tampered_label,
            max_samples=val_max,
        ) if base_val is not None else None

        test_ds = self._filter_segmentation_split(
            base_test,
            cache_key="test_tampered",
            tampered_label=tampered_label,
            max_samples=test_max,
        ) if base_test is not None else None

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
