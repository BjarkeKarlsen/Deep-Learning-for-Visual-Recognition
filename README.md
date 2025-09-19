# Deepfake Detection CNN

CNN for deepfake detection using SID_Set dataset.

## Setup

```bash
# Create environment
conda create -n deepfake-detection python=3.10
conda activate deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
# Train model
python main.py --train

# Evaluate model
python main.py --eval
```