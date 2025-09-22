# Deepfake Detection CNN
CNN for deepfake detection using SID_Set dataset.
3-class deepfake detection: real, synthetic, and tampered images.

For course: Deep Learning for Visual Recognition | Group 3.

**Note:** Currently downloads the entire SID_Set dataset to `~/.cache/huggingface/datasets/`. To use streaming mode (no download), modify `dataset.py` to use `load_dataset(..., streaming=True)`.

## Setup

```bash
# Create environment
conda create -n deepfake-env python
conda activate deepfake-env

# Install dependencies
pip install -r requirements.txt
```

## Configure
`config.py` - Configuration hub for hyperparameters and settings

## Run

```bash
# Train model
python main.py --train

# Evaluate model
python main.py --eval
```