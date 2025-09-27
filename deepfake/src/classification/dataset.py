import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Resize, Grayscale, Normalize, ToImage, ToDtype
from PIL import Image
from torch.utils.data import Dataset
    

class SIDClassificationDataset(Dataset):
    """
    Map-style dataset that standardises SID samples into tensors.

    Returns a dictionary with keys:
        - "image": float tensor normalised to the configured mean/std
        - "mask": float tensor (dummy zeros when the source lacks a mask)
        - "label": long tensor with the class index
    """

    @staticmethod
    def to_rgb(img):
        """Convert PIL image to RGB if needed."""
        return img.convert("RGB") if isinstance(img, Image.Image) else img

    
    def __init__(self,
                 dataset,
                 *,
                 image_size,
                 normalize_mean,
                 normalize_std,
                 transform=None,
                 transform_mask=None):
        self.dataset        = dataset
        self.image_size     = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std  = normalize_std

        # Default to the segmentation-style preprocessing so outputs stay aligned across tasks.
        if transform is None:
            self.transform = Compose([
                self.to_rgb,
                Resize((self.image_size, self.image_size)),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=normalize_mean, std=normalize_std)
            ])
        else:
            self.transform = transform

        if transform_mask is None:
            self.transform_mask = Compose([
                Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                Grayscale(num_output_channels=1),  # Keep mask tensors compatible with the segmentation pipeline
                ToImage(),
                ToDtype(torch.float32, scale=True)
            ])
        else:
            self.transform_mask = transform_mask
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.dataset[idx]

        image = self.transform(example["image"]) if self.transform else example["image"]
        assert image.ndim == 3 and image.shape[1:] == (self.image_size, self.image_size), (
            f"Image has wrong shape: {image.shape}"
        )

        mask_tensor = None
        if self.transform_mask:
            raw_mask = example.get("mask")
            if raw_mask is not None:
                mask_tensor = self.transform_mask(raw_mask)
            else:
                mask_tensor = torch.zeros(
                    (1, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            assert mask_tensor.ndim == 3 and mask_tensor.shape[1:] == (self.image_size, self.image_size), (
                f"Mask has wrong shape: {mask_tensor.shape}"
            )

        label = torch.tensor(example["label"], dtype=torch.long)

        sample = {"image": image, "label": label}
        if mask_tensor is not None:
            # Classification still exposes a mask tensor—zeros when missing—so
            # downstream utilities can treat both tasks uniformly.
            sample["mask"] = mask_tensor
        return sample
