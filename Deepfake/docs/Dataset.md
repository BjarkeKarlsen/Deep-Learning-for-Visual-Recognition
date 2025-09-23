# DATASET.md

# SIDDataset

Custom PyTorch `Dataset` for images, masks, and labels with preprocessing and optional device placement.

## Constructor Signature
```python
def __init__(
    self,
    dataset,
    *,
    image_size,
    normalize_mean,
    normalize_std,
    device=None,
    transform=None,
    transform_mask=None
)
```

- `dataset`: HF `Dataset` split object
- `image_size`: int, target height & width
- `normalize_mean` / `normalize_std`: lists of 3 floats
- `device`: `torch.device` (e.g., `'cpu'` or `'cuda'`)
- `transform`: override default image transform pipeline
- `transform_mask`: override default mask transform

## Default Transforms

- **Image**: `to_rgb` → `Resize(image_size)` → `ToImage` → `ToDtype(torch.float32)` → `Normalize(mean, std)`
- **Mask**: `Resize(image_size)` → `Grayscale(1)` → `ToImage` → `ToDtype(torch.float32)`

## Multiprocessing Compatibility

- Keyword-only parameters prevent pickling errors when using `num_workers > 0`.

## __getitem__ Behavior

1. Loads sample `ex = dataset[idx]`.
2. Applies `transform` to `ex['image']`.
3. Applies `transform_mask` if `ex['mask']` exists, else creates a zero mask.
4. Converts `ex['label']` to `torch.long` tensor.
5. Moves tensors to `device` if provided.
6. Returns a tuple `(image, label)`.

## Usage Example

```python
from dataset import SIDDataset
from torch.utils.data import DataLoader
import torch

# Assume train_ds is an HF Dataset
dataset = SIDDataset(
    train_ds,
    image_size=512,
    normalize_mean=[0.485,0.456,0.406],
    normalize_std=[0.229,0.224,0.225],
    device=torch.device('cuda')
)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
```

This file explains how to customize transforms, handle missing masks, and work with multi-worker loading without pickle issues.