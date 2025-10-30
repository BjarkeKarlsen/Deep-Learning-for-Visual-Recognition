#!/usr/bin/env python3
"""Generate class balance visualisations for the SID dataset splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from deepfake.config import Config
from deepfake.utils.config_loader import ConfigLoader
from deepfake.data.dataset_manager import SIDDatasetManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot class balance for SID dataset splits."
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split to analyse (default: train).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to inspect (useful for streaming).",
    )
    parser.add_argument(
        "--use-streaming",
        action="store_true",
        help="Enable Hugging Face streaming mode (requires --max-samples).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory where summary files and plots will be written.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config to resolve class names and data settings.",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> Config:
    loader = ConfigLoader(config_path=config_path)
    return loader.get_config()


def collect_samples(
    cfg: Config,
    split: str,
    max_samples: int | None,
    use_streaming: bool,
) -> Iterable[Dict]:
    manager = SIDDatasetManager(
        dataset_name=cfg.data.dataset_name,
        use_streaming=use_streaming,
    )

    dataset = manager.get_split(
        split_type=split,
        max_samples=max_samples,
    )
    return dataset


def summarise_labels(
    dataset: Iterable[Dict],
    label_mapping: Dict[int, str],
) -> Tuple[Counter, Counter]:
    counts = Counter()
    tamper_pixels = Counter()

    for example in dataset:
        label = example.get("label")
        if label is None:
            continue
        counts[label_mapping.get(label, str(label))] += 1

        mask = example.get("mask")
        if mask is not None:
            # SUMMARISE HOW MUCH OF EACH IMAGE IS TAMPERED FOR EXTRA CONTEXT.
            tamper_fraction = float(np.mean(np.array(mask) > 0))
            bucket = f"{int(tamper_fraction * 100):02d}%"
            tamper_pixels[bucket] += 1

    return counts, tamper_pixels


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary_json(
    output_dir: Path,
    split: str,
    class_counts: Counter,
    tamper_pixels: Counter,
) -> Path:
    summary_path = output_dir / f"class_balance_{split}.json"
    payload = {
        "split": split,
        "total_samples": int(sum(class_counts.values())),
        "class_counts": dict(class_counts),
        "tamper_pixel_buckets": dict(tamper_pixels),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return summary_path


def plot_class_balance(
    output_dir: Path,
    split: str,
    class_counts: Counter,
) -> Path:
    labels = list(class_counts.keys())
    totals = np.array([class_counts[label] for label in labels], dtype=float)
    percentages = 100.0 * totals / totals.sum() if totals.sum() else np.zeros_like(totals)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, totals, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title(f"SID {split.title()} Split Class Balance")
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Class")
    ax.set_ylim(0, max(totals) * 1.15 if totals.size else 1)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{pct:.1f}%\n({int(height)})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    plot_path = output_dir / f"class_balance_{split}.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def plot_tamper_distribution(
    output_dir: Path,
    split: str,
    tamper_pixels: Counter,
) -> Path | None:
    if not tamper_pixels:
        return None

    # Sort bucket labels numerically ("05%", "10%", ...)
    def _bucket_key(label: str) -> int:
        try:
            return int(label.rstrip("%"))
        except ValueError:
            return 0

    buckets = sorted(tamper_pixels.keys(), key=_bucket_key)
    values = np.array([tamper_pixels[bucket] for bucket in buckets], dtype=float)
    total = values.sum()

    # Guard against division by zero if counters are somehow empty.
    percentages = 100.0 * values / total if total else np.zeros_like(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(buckets, values, color="#dd8452")
    ax.set_title(f"SID {split.title()} Split Tamper-Area Distribution")
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Percent of Pixels Tampered")
    ax.set_ylim(0, values.max() * 1.15 if values.size else 1)

    # Thin out tick labels so the x-axis stays readable on dense histograms.
    if buckets:
        max_ticks = 12
        step = max(1, len(buckets) // max_ticks)
        tick_positions = list(range(0, len(buckets), step))
        if tick_positions[-1] != len(buckets) - 1:
            tick_positions.append(len(buckets) - 1)
        tick_labels = [buckets[idx] for idx in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    plot_path = output_dir / f"tamper_distribution_{split}.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = ensure_output_dir(args.output_dir)

    if args.use_streaming and args.max_samples is None:
        raise ValueError("When --use-streaming is set you must also provide --max-samples.")

    dataset = collect_samples(
        cfg=cfg,
        split=args.split,
        max_samples=args.max_samples,
        use_streaming=args.use_streaming,
    )

    label_mapping = {idx: name for idx, name in enumerate(cfg.model.class_names)}
    class_counts, tamper_pixels = summarise_labels(dataset, label_mapping)

    summary_path = write_summary_json(output_dir, args.split, class_counts, tamper_pixels)
    plot_path = plot_class_balance(output_dir, args.split, class_counts)
    tamper_plot_path = plot_tamper_distribution(output_dir, args.split, tamper_pixels)

    print(f"Wrote JSON summary to {summary_path}")
    print(f"Wrote class balance plot to {plot_path}")
    if tamper_plot_path is not None:
        print(f"Wrote tamper distribution plot to {tamper_plot_path}")


if __name__ == "__main__":
    main()
