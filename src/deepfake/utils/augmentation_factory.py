from __future__ import annotations

from typing import Optional
import random

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
    Grayscale,
)
from torchvision.transforms import functional as F

from deepfake.config.schema import AugmentationConfig


class SegmentationJointTransform:
    """Apply paired geometric transforms to image/mask samples."""
    # KEEPS IMAGE AND MASK AUGMENTATIONS IN LOCKSTEP FOR SEGMENTATION TASKS.

    def __init__(
        self,
        image_size: int,
        augment_cfg: Optional[AugmentationConfig],
        *,
        is_train: bool,
    ):
        self.image_size = image_size
        self.augment_cfg = augment_cfg
        self.is_train = is_train

    def _resize_pair(self, image, mask):
        # RESIZE BOTH IMAGE AND MASK TO THE TARGET RESOLUTION.
        resized_image = F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        resized_mask = None
        if mask is not None:
            resized_mask = F.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
        return resized_image, resized_mask

    def __call__(self, image, mask=None):
        if mask is None:
            return self._resize_pair(image, mask)

        cfg = self.augment_cfg
        if cfg and getattr(cfg, "enable", False) and self.is_train:
            # TRAIN-TIME AUGMENTATIONS APPLY RANDOM CROPS AND FLIPS TO BOTH IMAGE AND MASK.
            if getattr(cfg, "random_resized_crop", False):
                i, j, h, w = RandomResizedCrop.get_params(
                    image,
                    scale=(cfg.scale_min, cfg.scale_max),
                    ratio=(0.75, 1.3333),
                )
                image = F.resized_crop(
                    image,
                    top=i,
                    left=j,
                    height=h,
                    width=w,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                mask = F.resized_crop(
                    mask,
                    top=i,
                    left=j,
                    height=h,
                    width=w,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.NEAREST,
                )
            else:
                image, mask = self._resize_pair(image, mask)

            if cfg.horizontal_flip_prob > 0 and random.random() < cfg.horizontal_flip_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)
        else:
            # IN EVAL MODE OR WHEN AUGMENTATION IS DISABLED, JUST RESIZE TO THE TARGET SHAPE.
            image, mask = self._resize_pair(image, mask)

        return image, mask


def _build_common_tail(normalize_mean, normalize_std, augment_cfg: AugmentationConfig | None) -> list:
    # SHARED POST-PROCESSING TO CONVERT PIL INPUTS INTO NORMALISED TENSORS.
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
    # BUILDS THE IMAGE-ONLY PIPELINE USED BY THE CLASSIFIER.
    ops = []

    if augment_cfg and augment_cfg.enable and is_train:
        # WHEN TRAINING, ADD RANDOM CROPS, FLIPS, COLOUR JITTER, AND OPTIONAL BLURRING.
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
        # DURING EVAL OR WHEN AUGMENTATION IS DISABLED, PERFORM A SIMPLE RESIZE.
        ops.append(Resize((image_size, image_size), antialias=True))

    ops.extend(_build_common_tail(normalize_mean, normalize_std, augment_cfg if is_train else None))
    return Compose(ops)


def build_segmentation_transforms(
    image_size: int,
    normalize_mean,
    normalize_std,
    augment_cfg: Optional[AugmentationConfig] = None,
    *,
    is_train: bool = False,
):
    """
    Return joint/image/mask transforms for segmentation samples.

    joint_transform receives PIL image & mask and applies identical geometric
    ops to each; the image/mask transforms finalise tensor conversion.
    """

    # CREATE MATCHED TRANSFORMS FOR IMAGE, MASK, AND JOINT AUGMENTATIONS.
    joint_transform = SegmentationJointTransform(
        image_size=image_size,
        augment_cfg=augment_cfg,
        is_train=is_train,
    )

    image_ops = []
    # IMAGE-ONLY AUGMENTATIONS (JITTER, BLUR) APPLY AFTER GEOMETRIC STEPS PRESERVE ALIGNMENT WITH MASKS.
    if augment_cfg and augment_cfg.enable and is_train:
        if any(
            value > 0
            for value in (
                augment_cfg.color_jitter_brightness,
                augment_cfg.color_jitter_contrast,
                augment_cfg.color_jitter_saturation,
                augment_cfg.color_jitter_hue,
            )
        ):
            image_ops.append(
                ColorJitter(
                    brightness=augment_cfg.color_jitter_brightness,
                    contrast=augment_cfg.color_jitter_contrast,
                    saturation=augment_cfg.color_jitter_saturation,
                    hue=augment_cfg.color_jitter_hue,
                )
            )
        if augment_cfg.gaussian_blur_prob > 0:
            image_ops.append(
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

    image_ops.extend(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    mask_transform = Compose(
        [
            Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            Grayscale(num_output_channels=1),
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    )

    return joint_transform, Compose(image_ops), mask_transform
