# DATASET.md

## SIDDataset

Custom PyTorch `Dataset` that adapts SID samples into tensors for the
classification pipeline.

### Constructor Signature
```python
def __init__(
    self,
    dataset,
    *,
    image_size,
    normalize_mean,
    normalize_std,
    transform=None,
    transform_mask=None,
)
```

- `dataset`: Hugging Face `Dataset` split object
- `image_size`: target height & width in pixels
- `normalize_mean` / `normalize_std`: 3-element lists used by
  `torchvision.transforms.v2.Normalize`
- `transform`: optional override for the default image transform pipeline
- `transform_mask`: optional override for the default mask transform

### Default Transforms

- **Image**: `to_rgb` → `Resize(image_size)` → `ToImage` →
  `ToDtype(torch.float32, scale=True)` → `Normalize(mean, std)`
- **Mask**: `Resize(image_size)` → `Grayscale(num_output_channels=1)` →
  `ToImage` → `ToDtype(torch.float32, scale=True)`

### `__getitem__`

1. Loads sample `ex = dataset[idx]`.
2. Applies `transform` to `ex["image"]`.
3. Applies `transform_mask` when `ex["mask"]` exists; otherwise creates a
   zero mask with shape `(1, image_size, image_size)`.
4. Converts `ex["label"]` to `torch.long`.
5. Returns a dictionary with keys `"image"`, `"label"`, and `"mask"` (the mask
   is present even when it contains only zeros), mirroring the structure used
   by the segmentation pipeline.

### Usage Example

```python
from dataset import SIDDataset
from torch.utils.data import DataLoader

# Assume train_ds is an HF Dataset
train_dataset = SIDDataset(
    train_ds,
    image_size=512,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
```

The segmentation pipeline wraps its own dataset (`TamperedSegmentationDataset`)
which enforces the presence of masks and uses nearest-neighbour resizing to keep
binary masks intact.
