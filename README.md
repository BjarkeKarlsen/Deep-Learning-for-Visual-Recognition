# Deepfake Detection CNN

Minimal CNN for SID_Set dataset (3 classes: real, synthetic, tampered).

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train model (300 samples, 8 epochs)
python main.py --train

# Evaluate model (200 test samples)
python main.py --eval
```

## Performance
- **70% accuracy** with only 300 training samples
- **97.7% F1** on synthetic images (best)
- **49.1% F1** on real images

## Files
```
deepfake/
├── main.py                  # Training & evaluation script
├── models/best_model.pth    # Trained model (70% acc)
├── results/evaluation.json  # Evaluation metrics
├── requirements.txt         # Dependencies (5 packages)
└── README.md               # This file
```

Dataset uses **streaming** to load only needed samples, not full 210k.