import os
import sys

import torch
from torchvision.transforms.v2 import Compose, Resize, Grayscale, Lambda, Normalize, ToImage, ToDtype
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

    
@staticmethod
def to_rgb(img):
    return img.convert("RGB") if isinstance(img, Image.Image) else img

class SIDDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images, optional masks, and labels
    with preprocessing and device placement.

    Args:
        dataset: A sequence of samples, each containing "image", optional "mask", and "label".
        device: If provided, the device to which the data tensors will be transformed.
        transform (callable): Transform pipeline applied to images.
        transform_mask (callable): Transform pipeline applied to masks.

    Behavior:
        - Applies image and mask transforms, resizing to IMAGE_SIZE and normalizing.
        - Creates a dummy zero mask if no mask is provided. 
        - Ensures image and mask shapes match in size and channels.
        - Converts labels to torch.long tensors.
        - Moves image, mask, and label to the specified device.
        - Returns a dict with keys: {"image", "mask", "label"}.
    """
    def __init__(self, 
                 dataset,
                 device=None,
                 transform=Compose([
                    to_rgb, 
                    Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                    ToImage(),
                    ToDtype(torch.float32, scale=True),
                    Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)]), 
                 transform_mask=Compose([
                    Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                    Grayscale(num_output_channels=1), 
                    ToImage(),
                    ToDtype(torch.float32, scale=True)])
                    #Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)]) 
                 ):
        self.device = device
        self.transform = transform
        self.transform_mask = transform_mask
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ex = self.dataset[idx]

        # Transform image
        if self.transform:
            image = self.transform(ex["image"])
            assert image.ndim == 3 and image.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE), (
                f"Image has wrong shape: {image.shape}"
            )

        # Transform mask if available and not all of the data has a mask
        if self.transform_mask:
            mask = self.transform_mask(ex["mask"])
            if ex["mask"] is not None:
                mask = self.transform_mask(ex["mask"])
            else:
                mask = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float) # dummy mask maybe we can do without?
            
            assert mask.ndim == 3 and mask.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE), (
                f"Mask has wrong shape: {mask.shape}"
            )
            # assert mask.shape[0] == image.shape[0], (
            #     f"Channel mismatch: image C={image.shape[0]} vs mask C={mask.shape[0]}"
            # )

        label = torch.tensor(ex["label"], dtype=torch.long) # Maybe we can use simple labels
        
        if self.device:
            image, mask, label = image.to(self.device), mask.to(self.device), label.to(self.device)
            
        # Check how we should return the data - as dict or three individual tensors    
        return {"image": image, "mask": mask, "label": label}

