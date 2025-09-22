import io
import os
import sys

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_NAME, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, VAL_SAMPLES

class SIDDataset(Dataset):
    def __init__(self, split='train', max_samples=100):
        print(f"Loading dataset {DATASET_NAME}...")

        ds = load_dataset(DATASET_NAME)

        if split == 'test':
            print(f"Creating test split from validation data...")
            self.dataset = ds['validation'].select(range(VAL_SAMPLES, min(VAL_SAMPLES + max_samples, len(ds['validation']))))
        elif split == 'validation':
            print(f"Creating validation split...")
            self.dataset = ds['validation'].select(range(min(max_samples, len(ds['validation']))))
        else:
            print(f"Creating training split with {max_samples} samples...")
            self.dataset = ds['train'].select(range(min(max_samples, len(ds['train']))))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        print(f"Loaded {len(self.dataset)} samples for {split} split")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        if not isinstance(image, Image.Image):
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            else:
                image = Image.fromarray(image) if hasattr(image, '__array__') else image

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)
        label = int(item['label'])

        return image, label