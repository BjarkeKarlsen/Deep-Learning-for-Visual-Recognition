from __future__ import annotations

from typing import Optional

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import (
    Compose,
    RandomResizedCrop,
    Resize,
    RandomHorizontalFlip,
    ColorJitter,
    RandomApply,
    GaussianBlur,
    ToImage,
    ToDtype,
    Normalize,
    RandomErasing,
)

from deepfake.config.schema import AugmentationConfig


def _build_common_tail(normalize_mean, normalize_std, augment_cfg: AugmentationConfig | None) -> list:
    ops = [
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=normalize_mean, std=normalize_std),
    ]
    if augment_cfg and augment_cfg.enable and augment_cfg.random_erasing_prob > 0:
        ops.append(
            RandomErasing(
                p=augment_cfg.random_erasing_prob,
                scale=(augment_cfg.random_erasing_scale_min, augment_cfg.random_erasing_scale_max),
                ratio=(augment_cfg.random_erasing_ratio_min, augment_cfg.random_erasing_ratio_max),
            )
        )
    return ops


def build_classification_transform(
    image_size: int,
    normalize_mean,
    normalize_std,
    augment_cfg: Optional[AugmentationConfig] = None,
    *,
    is_train: bool = False,
):
    """Return a torchvision.v2 Compose for classification samples."""
    ops = []

    if augment_cfg and augment_cfg.enable and is_train:
        if augment_cfg.random_resized_crop:
            ops.append(
                RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(augment_cfg.scale_min, augment_cfg.scale_max),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        else:
            ops.append(Resize((image_size, image_size), antialias=True))

        if augment_cfg.horizontal_flip_prob > 0:
            ops.append(RandomHorizontalFlip(p=augment_cfg.horizontal_flip_prob))

        if any(
            value > 0
            for value in (
                augment_cfg.color_jitter_brightness,
                augment_cfg.color_jitter_contrast,
                augment_cfg.color_jitter_saturation,
                augment_cfg.color_jitter_hue,
            )
        ):
            ops.append(
                ColorJitter(
                    brightness=augment_cfg.color_jitter_brightness,
                    contrast=augment_cfg.color_jitter_contrast,
                    saturation=augment_cfg.color_jitter_saturation,
                    hue=augment_cfg.color_jitter_hue,
                )
            )

        if augment_cfg.gaussian_blur_prob > 0:
            ops.append(
                RandomApply(
                    [
                        GaussianBlur(
                            kernel_size=3,
                            sigma=(
                                augment_cfg.gaussian_blur_sigma_min,
                                augment_cfg.gaussian_blur_sigma_max,
                            ),
                        )
                    ],
                    p=augment_cfg.gaussian_blur_prob,
                )
            )
    else:
        ops.append(Resize((image_size, image_size), antialias=True))

    ops.extend(_build_common_tail(normalize_mean, normalize_std, augment_cfg if is_train else None))
    return Compose(ops)
