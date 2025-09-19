import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .dataset import SIDDataset
from .model import SimpleCNN
from .utils import get_device, print_gpu_info, set_seed
from config import *

def train():
    set_seed(SEED)
    device = get_device()

    print("="*60)
    print("TRAINING MODE")
    print("="*60)
    print_gpu_info(device)
    print(f"Train samples: {TRAIN_SAMPLES}, Val samples: {VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print("="*60)

    train_dataset = SIDDataset(split='train', max_samples=TRAIN_SAMPLES)
    val_dataset = SIDDataset(split='validation', max_samples=VAL_SAMPLES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0

    for epoch in range(EPOCHS):
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (val_acc: {val_acc:.1f}%)")

    print(f"\nTraining complete! Best validation: {best_val_acc:.1f}%")

if __name__ == "__main__":
    train()