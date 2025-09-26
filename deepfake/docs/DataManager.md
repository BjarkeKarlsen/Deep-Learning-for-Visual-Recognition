# DATAMANAGER.md

## SIDDatasetManager

`SIDDatasetManager` centralises loading, splitting, optional streaming, and
custom caching for the Hugging Face `saberzl/SID_Set` dataset. The manager is
shared by both the classification and segmentation pipelines.

### Key Features

- **Hugging Face Cache Awareness** – honours `HF_DATASETS_CACHE` and `HF_HOME`
  before falling back to `~/.cache/huggingface/datasets`.
- **Custom Split Caching** – derived subsets are stored under
  `~/.cache/huggingface/datasets/custom_splits/SID/<split>/` when
  `use_disk_cache=True`.
- **Streaming Support** – when `use_streaming=True`, the manager uses the
  Hugging Face iterable interface and materialises only the requested number of
  samples for downstream processing.
- **Segmentation Filtering** – `get_segmentation_splits` filters to samples that
  include masks and whose label matches `tampered_label` from the config.

### Usage

```python
from src.common.dataset_manager import SIDDatasetManager

manager = SIDDatasetManager(
    dataset_name="saberzl/SID_Set",
    use_streaming=False,
    use_disk_cache=True,
)

train_ds, val_ds, test_ds = manager.get_splits(
    train_max=1000,
    val_max=200,
    test_max=200,
)

train_tampered, val_tampered, test_tampered = manager.get_segmentation_splits(
    tampered_label=2,
    train_max=400,
    val_max=80,
    test_max=80,
)
```

### Methods

- `load_dataset()` – fetches the dataset via `datasets.load_dataset`, respecting
  streaming settings and caching the base splits for reuse.
- `get_splits(train_max, val_max, test_max, use_official_test=False,
  val_offset=30000, test_offset=0)` – returns `(train, val, test)` subsets. When
  the dataset lacks a validation split the manager derives one from the end of
  the training split and handles streaming iterables transparently.
- `get_segmentation_splits(**kwargs)` – wraps `get_splits` and filters to
  tampered samples that provide masks. Useful for the U-Net pipeline.
- `get_cache_info()` – reports the active Hugging Face cache directory, custom
  split cache location, and any relevant environment variables.
- `clear_custom_cache()` – deletes only the custom split cache, preserving the
  original Hugging Face dataset downloads.

### Streaming Notes

- For iterative splits, requesting `train_max`, `val_max`, or `test_max` limits
  the number of records materialised in memory.
- Some downstream PyTorch utilities expect random-access datasets. The provided
  pipelines wrap streaming splits in `Dataset.from_list` to restore indexing
  semantics after the requested samples are buffered.
