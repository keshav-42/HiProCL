"""
Data augmentation utilities for contrastive learning.
"""

from typing import Callable
import torch
from torchvision import transforms
from PIL import Image


class TwoCropsTransform:
    """
    Take two random crops of one image as the query and key.

    This creates two different augmented views of the same image,
    which is essential for contrastive learning.
    """

    def __init__(self, base_transform: Callable):
        """
        Args:
            base_transform: The transformation pipeline to apply twice
        """
        self.base_transform = base_transform

    def __call__(self, x: Image.Image) -> tuple:
        """
        Apply the transformation twice to create two views.

        Args:
            x: Input PIL image

        Returns:
            Tuple of (view1, view2)
        """
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2


def get_simclr_augmentation(image_size: int = 224, s: float = 1.0) -> transforms.Compose:
    """
    Get SimCLR-style augmentation pipeline for histopathology images.

    Args:
        image_size: Target image size
        s: Strength of color distortion (default: 1.0)

    Returns:
        Composition of transforms
    """
    # Color jitter parameters
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s,
        contrast=0.8 * s,
        saturation=0.8 * s,
        hue=0.2 * s
    )

    # Data augmentation pipeline
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))],
            p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return data_transforms


def get_eval_augmentation(image_size: int = 224) -> transforms.Compose:
    """
    Get evaluation/inference augmentation pipeline (no random augmentations).

    Args:
        image_size: Target image size

    Returns:
        Composition of transforms
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return data_transforms


def get_train_transforms_with_two_views(image_size: int = 224, s: float = 1.0) -> TwoCropsTransform:
    """
    Get training transforms that return two augmented views.

    Args:
        image_size: Target image size
        s: Strength of color distortion

    Returns:
        TwoCropsTransform instance
    """
    base_transform = get_simclr_augmentation(image_size, s)
    return TwoCropsTransform(base_transform)
