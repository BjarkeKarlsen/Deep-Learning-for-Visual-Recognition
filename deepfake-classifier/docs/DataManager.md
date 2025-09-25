# DATAMANAGER.md

# SIDDatasetManager

`SIDDatasetManager` centralizes dataset loading, splitting, streaming, and on-disk caching for the HuggingFace SID_Set dataset.

## Key Features

- **HuggingFace Caching**: Always uses `~/.cache/huggingface/datasets` by default. Controlled via `HF_HOME` and `HF_DATASETS_CACHE` environment variables.
- **Custom Split Caching**: Derived subsets (train/val/test) are saved under:
  ```
  ~/.cache/huggingface/datasets/custom_splits/SID/<split>/
  ```
- **Streaming Support**: Set `use_streaming=True` to stream without full download.
- **On-Disk Cache Control**: Toggle with `use_disk_cache` parameter.

## Usage

```python
from dataset_manager import SIDDatasetManager

# Initialize
manager = SIDDatasetManager(
    dataset_name="saberzl/SID_Set",
    use_streaming=False,
    use_disk_cache=True
)

# Retrieve splits (train, val, test)
train_ds, val_ds, test_ds = manager.get_splits(
    train_max=1000,  # limit training samples
    val_max=200,     # limit validation samples
    test_max=200
)
```

## Methods

- `load_dataset()`
  - Downloads (or streams) full dataset splits.

- `get_splits(train_max, val_max, test_max, use_official_test=False, val_offset=30000, test_offset=0)`
  - Returns (`train_ds`, `val_ds`, `test_ds`) as HF `Dataset` objects.

- `get_cache_info()`
  - Returns a dict with:
    - `hf_datasets_cache`: root HF cache path
    - `custom_splits_cache`: path for custom splits
    - Environment variables: `HF_HOME`, `HF_DATASETS_CACHE`

- `clear_custom_cache()`
  - Deletes only the custom split caches, preserves the original dataset cache.

- `get_cache_size()`
  - Returns human-readable sizes for total HF cache and custom splits only.


<details>
<summary>Example: Streaming Mode</summary>

```python
manager_stream = SIDDatasetManager(
    dataset_name="saberzl/SID_Set",
    use_streaming=True,
    use_disk_cache=False
)
train_ds, val_ds, test_ds = manager_stream.get_splits(
    train_max=500, val_max=100, test_max=100
)
```

</details>
