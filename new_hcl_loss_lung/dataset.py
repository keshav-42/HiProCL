"""
Dataset module for LC25000 Histopathology Dataset with hierarchical label extraction.
"""

import os
from pathlib import Path
from typing import Tuple, Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class LC25000Dataset(Dataset):
    """
    LC25000 Histopathology Dataset with hierarchical label extraction.

    Expected directory structure:
        data_root/
            lung_aca/
                lungaca001.png
                lungaca002.png
                ...
            lung_scc/
                lungscc001.png
                ...
            colon_aca/
                colonaca001.png
                ...

    Labels extracted:
        - l0_label: Organ type (0=colon, 1=lung)
        - l1_label: Cancer subtype (unique integer per subtype)
        - l2_label: Instance ID (unique integer per image)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        file_list: Optional[list] = None
    ):
        """
        Args:
            root_dir: Root directory containing subdirectories for each class
            transform: Optional transform to be applied on images
            file_list: Optional list of file paths (for train/val/test splits)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Build mapping dictionaries
        self.organ_to_l0 = {'colon': 0, 'lung': 1}

        # Define all possible subtypes
        self.subtype_to_l1 = {
            'colon_aca': 0,
            'colon_benign': 1,
            'lung_aca': 2,
            'lung_benign': 3,
            'lung_scc': 4
        }

        # Collect all image paths
        if file_list is not None:
            self.image_paths = file_list
        else:
            self.image_paths = []
            for subtype_dir in self.root_dir.iterdir():
                if subtype_dir.is_dir():
                    for img_path in subtype_dir.glob('*.png'):
                        self.image_paths.append(img_path)
                    for img_path in subtype_dir.glob('*.jpeg'):
                        self.image_paths.append(img_path)
                    for img_path in subtype_dir.glob('*.jpg'):
                        self.image_paths.append(img_path)

        # Sort for reproducibility
        self.image_paths = sorted(self.image_paths)

        # Build instance ID mapping
        self.path_to_l2 = {path: idx for idx, path in enumerate(self.image_paths)}

    def _parse_labels(self, image_path: Path) -> Tuple[int, int, int]:
        """
        Extract hierarchical labels from image path.

        Args:
            image_path: Path object to the image

        Returns:
            Tuple of (l0_label, l1_label, l2_label)
        """
        # Get subtype from parent directory name
        subtype = image_path.parent.name.lower()

        # Extract organ from subtype
        if 'colon' in subtype:
            organ = 'colon'
        elif 'lung' in subtype:
            organ = 'lung'
        else:
            raise ValueError(f"Unknown organ type in subtype: {subtype}")

        # Get labels
        l0_label = self.organ_to_l0[organ]
        l1_label = self.subtype_to_l1.get(subtype, -1)

        if l1_label == -1:
            raise ValueError(f"Unknown subtype: {subtype}")

        l2_label = self.path_to_l2[image_path]

        return l0_label, l1_label, l2_label

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (image, (l0_label, l1_label, l2_label))
        """
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Extract labels
        l0_label, l1_label, l2_label = self._parse_labels(img_path)

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, (l0_label, l1_label, l2_label)

    def get_class_distribution(self) -> dict:
        """Get distribution of samples across classes."""
        l0_counts = {}
        l1_counts = {}

        for img_path in self.image_paths:
            l0, l1, _ = self._parse_labels(img_path)
            l0_counts[l0] = l0_counts.get(l0, 0) + 1
            l1_counts[l1] = l1_counts.get(l1, 0) + 1

        return {
            'organ_distribution': l0_counts,
            'subtype_distribution': l1_counts,
            'total_samples': len(self.image_paths)
        }
