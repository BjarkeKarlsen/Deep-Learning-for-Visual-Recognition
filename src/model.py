# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # shared encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        
        # classification head
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # segmentation head
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(16, 1, 2, stride=2) # TESTING with 3 channels output for mask should not be propbaly
        
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # encoder
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x_enc = self.pool(x)  # shape (B,32,H/4,W/4)
        
        # classification branch
        cls = self.adaptive_pool(x_enc)
        cls = cls.view(cls.size(0), -1)
        cls = self.relu(self.fc1(cls))
        cls = self.dropout(cls)
        class_logits = self.fc2(cls)  # (B,num_classes)
        
        # segmentation branch
        seg = self.relu(self.upconv1(x_enc))  # (B,16,H/2,W/2)
        seg = self.upconv2(seg)              # (B,3, H,  W) #Check testing comment
        # we will apply BCEWithLogitsLoss so no sigmoid here
        mask_logits = seg
        
        return class_logits, mask_logits
