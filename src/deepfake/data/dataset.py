import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Resize, Grayscale, Normalize, ToImage, ToDtype
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from datasets import IterableDataset as HFIterableDataset, Dataset as HFDataset
    

class SIDClassificationDataset(Dataset):
    """
    Adapt SID records into tensors compatible with the classification pipeline.

    Yields dictionaries containing:
        - "image": float tensor normalised to the configured mean/std
        - "mask": float tensor (zeroed when the source lacks a mask)
        - "label": long tensor holding the class index
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
                 transform_mask=None,
                 max_samples=None,
                 return_mask=False,
                 return_label=False
                 ):  # Add max_samples for streaming datasets
        self.dataset        = dataset
        self.image_size     = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std  = normalize_std
        self.transform      = transform
        self.transform_mask = transform_mask
        self.max_samples = max_samples 
        self.return_mask = return_mask
        self.return_label = return_label
        self.is_streaming = isinstance(dataset, (HFIterableDataset, IterableDataset))
        
        if self.is_streaming:
            print(f"Converting streaming dataset to Dataset object...")
            
            # Take the samples we need and convert to Dataset
            samples = []
            for i, sample in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                samples.append(sample)
            
            # Create a proper HF Dataset object from the samples
            self.dataset = HFDataset.from_list(samples)
            self.is_streaming = False
            print(f"Converted {len(samples)} samples to Dataset object")


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
                Grayscale(num_output_channels=1),  # Preserve single-channel masks for segmentation utilities
                ToImage(),
                ToDtype(torch.float32, scale=True)
            ])
        else:
            self.transform_mask = transform_mask
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> dict:
        """Return a transformed sample dict and inject zero masks when the source omits them."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.dataset[idx]

        image = self.transform(example["image"]) if self.transform else example["image"]
        assert image.ndim == 3 and image.shape[1:] == (self.image_size, self.image_size), (
            f"Image has wrong shape: {image.shape}"
        )



        label = torch.tensor(example["label"], dtype=torch.long)

        outputs = {"image": image}
        
         # Mask
        mask_tensor :torch.Tensor = None
        if self.return_mask:
            raw_mask = example.get("mask")
            if raw_mask is not None:
                mask_tensor = self.transform_mask(raw_mask)
            else:
                raise ValueError("Expected mask but sample has none")

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            assert mask_tensor.ndim == 3 and mask_tensor.shape[1:] == (self.image_size, self.image_size), (
                f"Mask has wrong shape: {mask_tensor.shape}"
            )
            
            outputs.update({"mask": mask_tensor})
         
        # Label
        if self.return_label:
            label = example.get("label")
            if label is None:
                raise ValueError("Expected label but sample has none")
            outputs.update({"label": torch.tensor(label, dtype=torch.long)})

        return outputs
