import os
import sys

import torch
from torchvision.transforms.v2 import Compose, Resize, Grayscale, Lambda, Normalize, ToImage, ToDtype
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    

class SIDDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images, optional masks, and labels
    with preprocessing applied.

    Args:
        dataset: A sequence of samples, each containing "image", optional "mask", and "label".
        transform (callable): Transform pipeline applied to images.
        transform_mask (callable): Transform pipeline applied to masks.

    Behavior:
        - Applies image and mask transforms, resizing to IMAGE_SIZE and normalizing.
        - Creates a dummy zero mask if no mask is provided. 
        - Ensures image and mask shapes match in size and channels.
        - Converts labels to torch.long tensors.
        - Returns a dict with keys: {"image", "mask", "label"}.
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

        # Set default transforms if not provided
        if transform is None:
            self.transform = Compose([
                self.to_rgb,
                Resize((self.image_size, self.image_size), antialias=True),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=normalize_mean, std=normalize_std)
            ])
        else:
            self.transform = transform
            
        if transform_mask is None:
            self.transform_mask = Compose([
                Resize((self.image_size, self.image_size), antialias=True),
                Grayscale(num_output_channels=1), 
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
        ex = self.dataset[idx]

        # Transform image
        if self.transform:
            image = self.transform(ex["image"])
            assert image.ndim == 3 and image.shape[1:] == (self.image_size, self.image_size), (
                f"Image has wrong shape: {image.shape}"
            )

        # Transform mask if available and not all of the data has a mask
        if self.transform_mask:
            if ex["mask"] is not None:
                mask = self.transform_mask(ex["mask"])
            else:
                mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32) # Create dummy mask for samples without mask data
            assert mask.ndim == 3 and mask.shape[1:] == (self.image_size, self.image_size), (
                f"Mask has wrong shape: {mask.shape}"
            )

        label = torch.tensor(ex["label"], dtype=torch.long) 

        # TODO: Change the way to load the data when the PR is done. To be {"image": image, "mask": mask, "label": label}
        return image, label
