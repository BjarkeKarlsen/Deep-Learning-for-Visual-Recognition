# Deepfake Detection CNN
**3-class deepfake detection** (Real, Synthetic, Tampered) using the SID_Set dataset  
For course: *Deep Learning for Visual Recognition | Group 3*


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
All defaults live in `src/config/default.yaml`. Edit the repository-level `config.yaml` to override anything — it now lists every available option.

Useful commands:

```bash
# Inspect the merged config (defaults + overrides)
python -m src.utils.config_loader

# Write a fresh copy of the default template
python -m src.utils.config_loader --dump-default --output my-config.yaml
# Add --force to overwrite an existing file
```

The configuration sections map directly to the dataclasses in `src/config/schema.py` (`data`, `loader`, `model`, `training`, `paths`). Results and weights paths are resolved relative to the project root unless you provide absolute paths.


***

## Run

```bash
# Train model (optional: --config path/to/experiment.yaml)
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
