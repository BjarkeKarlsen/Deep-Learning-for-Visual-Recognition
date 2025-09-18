#!/usr/bin/env python3
"""
Deepfake Detection CNN - Training & Evaluation
Usage:
  python main.py --train    # Train model
  python main.py --eval     # Evaluate model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import io
from itertools import islice
from tqdm import tqdm
import numpy as np
import json
import argparse
import os

# Dataset
class SIDDataset(Dataset):
    def __init__(self, split='train', max_samples=100):
        print(f"Loading {max_samples} samples from SID_Set ({split})...")
        streaming_dataset = load_dataset("saberzl/SID_Set", split=split, streaming=True)
        self.data = list(islice(streaming_dataset, max_samples))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 3)  # 3 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train():
    """Train the model"""
    # Configuration
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 20
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 0.001

    class_names = {0: 'real', 1: 'synthetic', 2: 'tampered'}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("TRAINING MODE")
    print("="*60)
    print(f"Device: {device}")
    print(f"Train samples: {TRAIN_SAMPLES}, Val samples: {VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print("="*60)

    # Data loaders
    train_dataset = SIDDataset(split='train', max_samples=TRAIN_SAMPLES)
    val_dataset = SIDDataset(split='validation', max_samples=VAL_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0

    # Training loop
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"→ Saved best model (val_acc: {val_acc:.1f}%)")

    print(f"\n✓ Training complete! Best validation: {best_val_acc:.1f}%")

def evaluate():
    """Evaluate the model"""
    TEST_SAMPLES = 200
    BATCH_SIZE = 32

    class_names = {0: 'Real', 1: 'Synthetic', 2: 'Tampered'}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("EVALUATION MODE")
    print("="*60)

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    # Load test data
    test_dataset = SIDDataset(split='validation', max_samples=TEST_SAMPLES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate
    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics
    print("\n" + "="*60)
    print(f"Overall Accuracy: {accuracy*100:.1f}%")
    print("\nPer-class Accuracy:")

    results = {'overall_accuracy': float(accuracy), 'per_class': {}}

    for i, name in class_names.items():
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean()
            results['per_class'][name] = float(class_acc)
            print(f"  {name:10s}: {class_acc*100:.1f}%")

    # Confusion matrix
    conf_matrix = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            conf_matrix[i, j] = ((all_labels == i) & (all_preds == j)).sum()

    print("\nConfusion Matrix:")
    print("         Pred: R   S   T")
    for i, name in enumerate(['Real    ', 'Synthetic', 'Tampered ']):
        print(f"{name}: {conf_matrix[i]}")

    # Save results
    os.makedirs('results', exist_ok=True)
    results['confusion_matrix'] = conf_matrix.tolist()

    with open('results/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to results/evaluation.json")

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection CNN')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--eval', action='store_true', help='Evaluate model')

    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        if not os.path.exists('models/best_model.pth'):
            print("Error: No trained model found. Run with --train first.")
            return
        evaluate()
    else:
        print("Usage: python main.py --train  OR  python main.py --eval")

if __name__ == '__main__':
    main()