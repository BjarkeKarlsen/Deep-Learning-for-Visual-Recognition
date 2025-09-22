# Deepfake Detection CNN
**3-class deepfake detection** (Real, Synthetic, Tampered) using the SID_Set dataset  
For course: *Deep Learning for Visual Recognition | Group 3*

For course: Deep Learning for Visual Recognition | Group 3.

**Note:** Currently downloads the entire SID_Set dataset to `~/.cache/huggingface/datasets/`. To use streaming mode (no download), modify `dataset.py` to use `load_dataset(..., streaming=True)`.

## Setup

```bash
# Create environment
conda create -n deepfake-env python=3.13.7
conda activate deepfake-env

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml` to override any defaults. Example:

```yaml
data:
  train_samples: 100
  val_samples: 20

loader:
  batch_size: 8
  shuffle_train: true
  num_workers: 4

training:
  epochs: 50
  learning_rate: 0.0005

paths:
  logging_dir: "logs/exp1"
```

The top‐level keys mirror `config.py` dataclasses: `data`, `loader`, `model`, `training`, `paths`.

Results and model weights will be saved under paths from `cfg.paths`.


***

## Run

```bash
# Train model
python main.py --train

# Evaluate model
python main.py --eval
```

## Information
For more documentation see docs under `Deepfake\docs`


## Notes

- Data is cached by default in `~/.cache/huggingface/datasets/`.  
- To enable streaming (no full download), set `cfg.data.use_streaming: true` in `config.yaml`.

***

## How the Data Manager Works

See **DATAMANAGER.md** for details on:

- HuggingFace cache locations  
- Custom split caching under `custom_splits/SID/`  
- Enabling streaming vs. full download  

***

## Dataset Class Details

See **DATASET.md** for information on:

- Keyword-only preprocessing parameters  
- Default transforms derived from config  
- Multiprocessing pickle fixes  

***
