from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import io
import os
import sys
from itertools import islice
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_NAME, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, VAL_SAMPLES

class SIDDataset(Dataset):
    def __init__(self, split='train', max_samples=100):
        if split == 'test':
            print(f"Loading test samples from {DATASET_NAME} (split from validation)...")
            streaming_dataset = load_dataset(DATASET_NAME, split='validation', streaming=True)
            self.data = list(islice(streaming_dataset, VAL_SAMPLES + max_samples))[VAL_SAMPLES:]
        elif split == 'validation':
            print(f"Loading validation samples from {DATASET_NAME}...")
            streaming_dataset = load_dataset(DATASET_NAME, split='validation', streaming=True)
            self.data = list(islice(streaming_dataset, max_samples))
        else:
            print(f"Loading {max_samples} samples from {DATASET_NAME} ({split})...")
            streaming_dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
            self.data = list(islice(streaming_dataset, max_samples))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item['image'], Image.Image):
            image = item['image']
        else:
            image = Image.open(io.BytesIO(item['image']))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)
        label = int(item['label'])
        return image, label