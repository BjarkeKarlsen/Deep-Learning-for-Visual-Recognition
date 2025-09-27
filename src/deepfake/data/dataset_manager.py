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
    """Coordinate SID dataset loading, custom split materialisation, and caching."""
    def __init__(
        self,
        dataset_name: str,
        use_disk_cache: bool = True,
        use_streaming: bool = False,
    ):
        self.dataset_name = dataset_name
        self.use_disk_cache = use_disk_cache
        if use_streaming:
            import warnings
            warnings.warn(
                "Streaming mode is temporarily disabled; falling back to map-style datasets to avoid Python shutdown issues.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.use_streaming = False
        self._streaming_requested = use_streaming
        self._cached_splits: Dict[str, Dataset] = {}

    def _ensure_indexable_split(self, split: str) -> Dataset:
        """Materialise streaming splits so downstream code retains random access."""
        ds = self._get_base_split(split)
        if isinstance(ds, IterableDataset):
            if self.use_streaming:
                raise ValueError(
                    f"Split '{split}' is in streaming mode; random access is unavailable. "
                    "Provide an explicit sample cap or disable streaming."
                )
            # Some HF snapshots still expose iterable shards; materialise once to preserve len/index/select semantics.
            materialised = self._slice_split(ds, max_samples=None)
            self._cached_splits[split] = materialised
            return materialised
        return ds

    @staticmethod
    def _empty_like(reference: Optional[Dataset]) -> Dataset:
        """Return an empty dataset sharing the same schema when possible."""
        if reference is not None:
            try:
                return reference.select([])
            except Exception:
                pass
        return Dataset.from_list([])

    @staticmethod
    def _normalise_streaming_cap(split: str, cap: Optional[int]) -> int:
        if cap is None:
            raise ValueError(
                f"Streaming mode requires a finite sample cap for the '{split}' split."
            )
        if cap < 0:
            raise ValueError(
                f"Streaming sample cap for split '{split}' must be non-negative (got {cap})."
            )
        return cap

    def _get_streaming_splits(
        self,
        *,
        train_max: Optional[int],
        val_max: Optional[int],
        test_max: Optional[int],
        test_offset: int,
        use_official_test: bool,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Materialise bounded train/val/test subsets while preserving streaming semantics."""

        splits = self.load_dataset()
        if "train" not in splits:
            raise ValueError("Streaming dataset does not provide a 'train' split.")

        if use_official_test and test_max is None:
            raise ValueError(
                "Streaming mode requires 'test_max' when 'use_official_test' is True."
            )

        train_cap = self._normalise_streaming_cap("train", train_max)
        train_ds = self._load_or_cache_split(
            "train_subset",
            max_samples=train_cap,
            derive_from="train",
            start=0,
        )

        val_cap = 0
        if "validation" in splits:
            if val_max is None:
                raise ValueError(
                    "Streaming mode requires 'val_max' when a validation split is available."
                )
            val_cap = self._normalise_streaming_cap("validation", val_max)
            val_ds = self._load_or_cache_split(
                "validation_subset",
                max_samples=val_cap,
                derive_from="validation",
                start=0,
            )
        else:
            if val_max not in (None, 0):
                raise ValueError(
                    "Streaming mode cannot derive a validation split without an original "
                    "'validation' split; consider enabling the official validation split "
                    "or disabling streaming."
                )
            val_ds = self._empty_like(train_ds)

        requested_test_cap = 0 if test_max is None else test_max
        test_ds: Dataset
        if requested_test_cap == 0:
            test_ds = self._empty_like(val_ds if "validation" in splits else train_ds)
        else:
            test_cap = self._normalise_streaming_cap("test", requested_test_cap)
            if use_official_test:
                if "test" not in splits:
                    raise ValueError(
                        "Requested official test split in streaming mode, but the dataset "
                        "does not provide a 'test' split."
                    )
                start_from_test = max(0, test_offset)
                test_ds = self._load_or_cache_split(
                    "test_subset",
                    max_samples=test_cap,
                    derive_from="test",
                    start=start_from_test,
                )
            else:
                if "validation" not in splits:
                    raise ValueError(
                        "Streaming mode requires a validation split to derive the test subset "
                        "when 'use_official_test' is False."
                    )
                if val_max is None:
                    raise ValueError(
                        "Streaming mode needs 'val_max' to derive the test subset from validation."
                    )
                start_from_validation = val_cap + max(0, test_offset)
                test_ds = self._load_or_cache_split(
                    "test_custom",
                    max_samples=test_cap,
                    derive_from="validation",
                    start=start_from_validation,
                )

        return train_ds, val_ds, test_ds

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
        """Compute half-open index ranges for validation and optional test subsets."""

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
        """Return a dataset view with the provided half-open intervals removed."""

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
        """Resolve the Hugging Face cache directory respecting env overrides."""
        # Honor environment overrides from most specific to least specific.
        hf_datasets_cache = os.environ.get('HF_DATASETS_CACHE')
        if hf_datasets_cache:
            return hf_datasets_cache

        hf_home = os.environ.get('HF_HOME')
        if hf_home:
            return os.path.join(hf_home, 'datasets')

        # Fall back to Hugging Face's conventional cache path.
        return os.path.expanduser('~/.cache/huggingface/datasets')

    def _cache_path(self, split: str, *, identifier: Optional[str] = None) -> str:
        """Generate a deterministic cache path scoped by dataset and split options."""
        hf_cache = self._get_hf_cache_dir()
        dataset_slug = self.dataset_name.replace("/", "__")
        parts = [split]
        if identifier:
            parts.append(identifier)
        cache_dir = os.path.join(hf_cache, "custom_splits", "SID", dataset_slug, "__".join(parts))
        return cache_dir

    def load_dataset(self) -> Dict[str, Dataset]:
        """Load the SID dataset (streaming or map-style) and memoize its splits locally."""
        if not self._cached_splits:
            print(f"Loading dataset {self.dataset_name} from Hugging Face...")

            full = load_dataset(
                self.dataset_name,
                streaming=self.use_streaming,
            )

            for split, ds in full.items():
                if isinstance(ds, IterableDataset) and not self.use_streaming:
                    # HF may return iterable shards even from local cache; materialise once to restore random access APIs.
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
        """Return a contiguous slice, materialising iterables when streaming semantics require it."""
        if isinstance(ds, IterableDataset):
            iterator = ds
            if start:
                if hasattr(iterator, "skip"):
                    iterator = iterator.skip(start)
                else:
                    iterator = islice(iterator, start, None)

            if max_samples is None:
                if self.use_streaming:
                    raise ValueError(
                        "Streaming splits require an explicit sample cap; "
                        "provide max_samples when requesting materialisation."
                    )
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
        """Load or derive a split and optionally persist it inside the HF cache hierarchy."""
        if self.use_streaming and max_samples is None:
            raise ValueError(
                "Streaming mode requires 'max_samples' to be specified when "
                f"loading the {derive_from or split} split."
            )

        id_components = [derive_from or split]
        max_str = str(max_samples) if max_samples is not None else "all"
        id_components.append(f"max{max_str}")
        if start:
            id_components.append(f"start{start}")
        if self.use_streaming:
            id_components.append("stream")
        identifier = "__".join(id_components)

        path = self._cache_path(split, identifier=identifier)
        
        # Step 1: reuse any cached materialisation for derived splits when present.
        if (
            self.use_disk_cache
            and derive_from is not None  # Only custom splits are cached
            and os.path.isdir(path)
            and os.listdir(path)
        ):
            print(f"Loading {split} split from HF cache: {path}")
            return load_from_disk(path)

        # Step 2: build the split from its source dataset.
        if derive_from:
            base = self._get_base_split(derive_from)
            ds = self._slice_split(base, max_samples, start)
        else:
            base = self._get_base_split(split)
            ds = self._slice_split(base, max_samples)

        # Step 3: cache custom splits for reuse in future runs.
        if derive_from and self.use_disk_cache:
            try:
                length = len(ds)
            except TypeError:
                # Streaming datasets may not expose len(); sample into a list once to compute it for logging/cache checks.
                tmp = list(ds.take(max_samples or 1))
                length = len(tmp)

            if length > 0:
                # Create the parent directory before persisting the derived split.
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
        """Return train/validation/test splits, caching any derived subsets for reuse."""
        if self.use_streaming:
            return self._get_streaming_splits(
                train_max=train_max,
                val_max=val_max,
                test_max=test_max,
                test_offset=test_offset,
                use_official_test=use_official_test,
            )

        splits = self.load_dataset()
        has_validation = "validation" in splits
        use_official_test_split = use_official_test and "test" in splits

        # Prepare validation/test holdouts straight from the dedicated validation split when available.
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
                    test_ds = None  # Populated later if we switch to the official test split.
            else:
                # With explicit caps, subset validation directly without running holdout math.
                val_ds = self._load_or_cache_split(
                    "validation_subset",
                    max_samples=val_max,
                    derive_from="validation",
                    start=0,
                )
                test_ds = None
            # Training samples come straight from the HF train split unless capped.
            if train_max is None:
                train_ds = self._ensure_indexable_split("train")
            else:
                train_ds = self._load_or_cache_split(
                    "train_subset", max_samples=train_max, derive_from="train"
                )

        else:
            # Fall back to carving validation/test holdouts out of the train split tail.
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

        # Optionally swap in the official test split, respecting any provided cap.
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
        """Filter a split to tampered samples and optionally persist the result."""
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
            if max_samples is not None:
                collected_indices = []
                for idx, example in enumerate(ds):
                    if _has_mask_with_label(example, target_label=tampered_label):
                        collected_indices.append(idx)
                        if len(collected_indices) == max_samples:
                            break
                filtered = ds.select(collected_indices) if collected_indices else ds.select([])
            else:
                filtered = ds.filter(
                    _has_mask_with_label,
                    fn_kwargs={"target_label": tampered_label},
                )

        if self.use_disk_cache:
            if len(filtered) > 0:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                print(f"Caching custom {cache_key} split to HF cache: {cache_path}")
                filtered.save_to_disk(cache_path)
            else:
                print(f"Skipping cache for empty {cache_key} split")

        return filtered

    def close(self) -> None:
        """Release cached streaming datasets so background workers shut down cleanly."""
        if self.use_streaming and self._cached_splits:
            self._cached_splits.clear()
            try:
                import gc
                gc.collect()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

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
            train_max=train_max,
            val_max=val_max,
            test_max=test_max,
            val_offset=val_offset,
            test_offset=test_offset,
            use_official_test=use_official_test,
        )

        # Retain only samples with a tamper mask so GPUs are not wasted on negatives with no pixel supervision signal.
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
        """Summarise active cache directories along with relevant environment overrides."""
        hf_cache = self._get_hf_cache_dir()
        custom_splits_cache = os.path.join(hf_cache, "custom_splits", "SID")
        
        info = {
            "hf_datasets_cache": hf_cache,
            "custom_splits_cache": custom_splits_cache,
        }
        
        # Include environment overrides for easier cache-debugging downstream.
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
