"""
Model architectures for hierarchical contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import timm


class Backbone(nn.Module):
    """
    CTransPath backbone for feature extraction from histopathology images.
    """

    def __init__(self, pretrained_path: str = None, feature_dim: int = 768):
        """
        Args:
            pretrained_path: Path to CTransPath pretrained weights (.pth file)
            feature_dim: Dimension of output features (768 for swin_tiny)
        """
        super(Backbone, self).__init__()

        # Load CTransPath model (Swin Transformer Tiny)
        self.model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=0)
        self.feature_dim = feature_dim

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, pretrained_path: str):
        """
        Load pretrained CTransPath weights.

        Args:
            pretrained_path: Path to the .pth file
        """
        if not Path(pretrained_path).exists():
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Initializing with random weights instead.")
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v

            # Load weights (ignore head if present)
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded CTransPath pretrained weights from {pretrained_path}")
            if msg.missing_keys:
                print(f"Missing keys: {msg.missing_keys[:5]}...")  # Show first 5
            if msg.unexpected_keys:
                print(f"Unexpected keys: {msg.unexpected_keys[:5]}...")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Initializing with random weights instead.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Feature vectors [B, feature_dim]
        """
        return self.model(x)


class ProjectionHead(nn.Module):
    """
    Projection head (2-layer MLP) for contrastive learning.

    Projects backbone features to a lower-dimensional space with L2 normalization.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, output_dim: int = 128):
        """
        Args:
            input_dim: Input feature dimension (from backbone)
            hidden_dim: Hidden layer dimension
            output_dim: Output projection dimension
        """
        super(ProjectionHead, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.

        Args:
            x: Input features [B, input_dim]

        Returns:
            L2-normalized projected features [B, output_dim]
        """
        x = self.projection(x)
        # L2 normalization
        x = F.normalize(x, dim=1, p=2)
        return x


class HCLModel(nn.Module):
    """
    Complete Hierarchical Contrastive Learning model.

    Combines backbone and projection head.
    """

    def __init__(
        self,
        pretrained_path: str = None,
        backbone_dim: int = 768,
        projection_hidden_dim: int = 2048,
        projection_output_dim: int = 128
    ):
        """
        Args:
            pretrained_path: Path to CTransPath pretrained weights
            backbone_dim: Backbone output dimension
            projection_hidden_dim: Projection head hidden dimension
            projection_output_dim: Projection head output dimension
        """
        super(HCLModel, self).__init__()

        self.backbone = Backbone(pretrained_path, backbone_dim)
        self.projection_head = ProjectionHead(
            backbone_dim,
            projection_hidden_dim,
            projection_output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through full model.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Projected and normalized features [B, output_dim]
        """
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from backbone only (for linear probing).

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Backbone features [B, backbone_dim]
        """
        return self.backbone(x)
