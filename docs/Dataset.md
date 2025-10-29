# DATASET.md

## `SIDClassificationDataset`

`SIDClassificationDataset` adapts SID samples into tensors that work for both
the classification and segmentation pipelines. The class now exposes optional
flags that control whether masks and/or labels are returned, so it doubles as
the segmentation dataset.

### Constructor

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
    joint_transform=None,
    max_samples=None,
    return_mask=False,
    return_label=False,
)
```

- `dataset`: Hugging Face `Dataset` or `IterableDataset` produced by
  `SIDDatasetManager.get_split`.
- `image_size`: final height/width in pixels.
- `normalize_mean` / `normalize_std`: 3-element lists passed to
  `torchvision.transforms.v2.Normalize`.
- `transform`: image-only pipeline; defaults to resize → tensor → normalise.
- `transform_mask`: mask pipeline; defaults to nearest-neighbour resize → tensor.
- `joint_transform`: optional callable that receives `(image, mask)` and must
  return the transformed pair (used for segmentation augmentations).
- `max_samples`: reserved for future streaming support (currently unused).
- `return_mask`: include the transformed mask in the output; raises if a sample
  has no mask.
- `return_label`: include the class label (`torch.long`); raises if missing.

### Default Output

Depending on the flags, `__getitem__` returns a dictionary containing:

- `"image"` – tensor of shape `(3, H, W)` scaled to `[0, 1]` and normalised.
- `"mask"` – tensor of shape `(1, H, W)` with binary values (only when
  `return_mask=True` and the source sample provides a mask).
- `"label"` – tensor scalar of type `torch.long` (only when
  `return_label=True`).

When `return_mask=True` and the underlying dataset lacks a mask for a sample,
the dataset raises `ValueError`. This keeps segmentation training/evaluation
strict.

### Usage Examples

**Classification loader**

```python
from torch.utils.data import DataLoader
from deepfake.data import SIDDatasetManager, TRAIN
from deepfake.data.dataset import SIDClassificationDataset
from deepfake.utils.augmentation_factory import build_classification_transform

manager = SIDDatasetManager(dataset_name="saberzl/SID_Set")
train_split = manager.get_split(split_type=TRAIN, max_samples=256)

transform = build_classification_transform(
    image_size=224,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    augment_cfg=None,
    is_train=True,
)

train_dataset = SIDClassificationDataset(
    train_split,
    image_size=224,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    transform=transform,
    return_label=True,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

**Segmentation loader**

```python
from deepfake.data.dataset_manager import DatasetFilters, TRAIN
from deepfake.data.dataset import SIDClassificationDataset
from deepfake.utils.augmentation_factory import build_segmentation_transforms

train_split = manager.get_split(
    split_type=TRAIN,
    max_samples=128,
    filter_fn=DatasetFilters.tampered_with_masks,
)

joint_tf, image_tf, mask_tf = build_segmentation_transforms(
    image_size=256,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    augment_cfg=None,
    is_train=True,
)

seg_dataset = SIDClassificationDataset(
    train_split,
    image_size=256,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    transform=image_tf,
    transform_mask=mask_tf,
    joint_transform=joint_tf,
    return_mask=True,
    return_label=False,
)
```

The segmentation trainer uses exactly this pattern to ensure masks and images
stay aligned under joint augmentations.
