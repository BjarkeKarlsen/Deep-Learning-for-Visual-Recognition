# DATAMANAGER.md

## `SIDDatasetManager`

`SIDDatasetManager` is a thin wrapper around `datasets.load_dataset` that
standardises how the toolkit materialises train/validation/test splits from the
[`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) dataset.
Both the classification and segmentation pipelines build their dataloaders on
top of it.

### What It Does Today

- Loads individual splits through `get_split(...)`.
- Optionally materialises streaming splits into an in-memory Hugging Face
  `Dataset` when `use_streaming=True`.
- Applies simple filtering callbacks (e.g. `DatasetFilters.tampered_with_masks`)
  before slicing down to the requested sample count.
- Derives test splits from validation/train when the dataset does not expose a
  dedicated `test` split.

⚠️ **No custom caching**: earlier iterations wrote derived splits into a custom
cache directory. That behaviour has been removed, and the current manager does
not consult `data.use_disk_cache`.

⚠️ **No multi-split helper**: `get_splits` / `get_segmentation_splits` no longer
exist. The pipelines call `get_split` separately for each split they need.

### Constructor

```python
def __init__(
    self,
    dataset_name: str,
    use_streaming: bool = False,
    cache_dir: Optional[str] = None,
    download_mode: DownloadMode = DownloadMode.REUSE_CACHE_IF_EXISTS,
)
```

- `dataset_name`: Hugging Face dataset identifier (defaults to
  `"saberzl/SID_Set"` in the config).
- `use_streaming`: whether to use HF's streaming mode. When `True`,
  `get_split(..., max_samples=N)` must supply `max_samples` so the generator can
  be buffered into memory.
- `cache_dir`: optional Hugging Face cache override.
- `download_mode`: forwarded directly to `datasets.load_dataset`.

### `get_split(...)`

```python
manager.get_split(
    split_type="train",           # "train" | "validation" | "test"
    max_samples=10,               # Optional cap (required for streaming)
    use_test_or_val_as_test_set=False,
    val_offset=0,
    train_offset=0,
    filter_fn=DatasetFilters.classification_only,
)
```

Key points:

- When `split_type="test"` and the dataset lacks a formal test split, set
  `use_test_or_val_as_test_set=True` to derive a hold-out window from the end of
  the validation or training split (controlled via `val_offset` /
  `train_offset`).
- `filter_fn` receives individual HF examples and should return `True` to keep
  them. The segmentation pipeline uses
  `DatasetFilters.tampered_with_masks` to drop samples without masks.
- When `use_streaming=True`, the manager buffers `max_samples * 4` examples
  before filtering to improve the odds of collecting enough filtered records.
  If fewer than `max_samples` survive the filter the manager logs a warning and
  returns the smaller slice.

### Example Usage

```python
from datasets import DownloadMode
from deepfake.data.dataset_manager import SIDDatasetManager, DatasetFilters, TRAIN, VALIDATION

manager = SIDDatasetManager(
    dataset_name="saberzl/SID_Set",
    use_streaming=False,
    download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
)

train_ds = manager.get_split(
    split_type=TRAIN,
    max_samples=100,
)
val_ds = manager.get_split(
    split_type=VALIDATION,
    max_samples=40,
)
seg_train = manager.get_split(
    split_type=TRAIN,
    max_samples=200,
    filter_fn=DatasetFilters.tampered_with_masks,
)
```

### Streaming Notes

- Set `data.use_streaming: true` in your YAML config to switch the pipelines
  into streaming mode. You **must** keep `data.train_samples`,
  `data.val_samples`, and `data.test_samples` finite; otherwise `Dataset.take`
  cannot materialise the stream.
- The manager converts the buffered generator into a regular HF `Dataset`
  instance so downstream PyTorch `DataLoader`s retain random-access semantics.
