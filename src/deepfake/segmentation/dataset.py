import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Grayscale, Normalize, Resize, ToDtype, ToImage


class TamperedSegmentationDataset(Dataset):
    """Dataset wrapper that yields only samples with tamper masks."""

    @staticmethod
    def _ensure_rgb(img):
        return img.convert("RGB") if isinstance(img, Image.Image) else img

    def __init__(
        self,
        dataset,
        *,
        image_size,
        normalize_mean,
        normalize_std,
        transform=None,
        mask_transform=None,
        return_label: bool = False,
    ):
        self.dataset = dataset
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.return_label = return_label

        if transform is None:
            self.image_transform = Compose([
                self._ensure_rgb,
                Resize((self.image_size, self.image_size)),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=normalize_mean, std=normalize_std),
            ])
        else:
            self.image_transform = transform

        if mask_transform is None:
            self.mask_transform = Compose([
                Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                Grayscale(num_output_channels=1),  # Ensure masks stay single-channel for the binary tamper task
                ToImage(),
                ToDtype(torch.float32, scale=True),
            ])
        else:
            self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        mask = example.get("mask")
        if mask is None:
            raise ValueError("Segmentation dataset received a sample without a mask")

        image = self.image_transform(example["image"])
        mask_tensor = self.mask_transform(mask)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        if self.return_label:
            label = torch.tensor(example["label"], dtype=torch.long)
            return image, mask_tensor, label
        return image, mask_tensor
