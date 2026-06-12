"""
Baseline Loss Functions for Comparison

This module implements standard contrastive and supervised losses for benchmarking
against the custom Hierarchical Contrastive Loss:

1. MoCo (Momentum Contrastive Learning)
2. SupCon (Supervised Contrastive Learning)
3. CrossEntropy (Standard supervised classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoLoss(nn.Module):
    """01






































































































































































































































    MoCo: Momentum Contrastive Learning

    Reference: "Momentum Contrast for Unsupervised Visual Representation Learning"
    He et al., CVPR 2020

    Key features:
    - Momentum encoder for generating keys
    - Queue of negative samples for memory bank
    - Instance discrimination objective
    """

    def __init__(
        self,
        temperature: float = 0.07,
        queue_size: int = 4096,
        momentum: float = 0.999,
        feature_dim: int = 128,
        eps: float = 1e-8
    ):
        """
        Args:
            temperature: Temperature for softmax
            queue_size: Size of the negative sample queue
            momentum: Momentum for updating key encoder
            feature_dim: Dimension of feature vectors
            eps: Small epsilon for numerical stability
        """
        super(MoCoLoss, self).__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        self.eps = eps

        # Create queue for negative samples
        self.register_buffer('queue', torch.randn(feature_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update queue with new keys.

        Args:
            keys: [B, D] new key features to add
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest samples
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining

        self.queue_ptr[0] = ptr

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """
        Compute MoCo loss.

        Args:
            features: [2B, D] normalized features (query and key concatenated)
            labels: [2B, 3] hierarchical labels (not used for MoCo)

        Returns:
            Dictionary with loss components
        """
        batch_size = features.shape[0] // 2

        # Split into queries and keys
        queries = features[:batch_size]  # [B, D]
        keys = features[batch_size:]     # [B, D]

        # Positive logits: [B, 1]
        l_pos = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1)

        # Negative logits: [B, K]
        l_neg = torch.einsum(
            'nc,ck->nk', [queries, self.queue.clone().detach()])

        # Logits: [B, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temperature

        # Labels: positives are at index 0
        labels_ce = torch.zeros(
            batch_size, dtype=torch.long, device=features.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels_ce)

        # Update queue
        self._dequeue_and_enqueue(keys)

        return {
            'loss': loss,
            'moco_loss': loss.item()
        }


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss

    Reference: "Supervised Contrastive Learning"
    Khosla et al., NeurIPS 2020

    Key features:
    - Uses label information to define positive pairs
    - Pulls together samples from same class
    - Pushes apart samples from different classes
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07,
        eps: float = 1e-8
    ):
        """
        Args:
            temperature: Temperature for softmax
            contrast_mode: 'all' or 'one' (use all positives or one)
            base_temperature: Base temperature for normalization
            eps: Small epsilon for numerical stability
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute supervised contrastive loss.

        Args:
            features: [2B, D] normalized features (two views concatenated)
            labels: [2B, 3] hierarchical labels - uses L1 (subtype) for supervision

        Returns:
            Dictionary with loss components
        """
        device = features.device
        batch_size = features.shape[0]

        # Extract L1 labels (subtype level)
        labels = labels[:, 1].contiguous().view(-1, 1)  # [2B, 1]

        # Create mask for positive pairs (same label)
        mask = torch.eq(labels, labels.T).float().to(device)  # [2B, 2B]

        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )  # [2B, 2B]

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )  # [2B, 2B]

        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - \
            torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return {
            'loss': loss,
            'supcon_loss': loss.item()
        }


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Classification Loss

    This serves as a supervised baseline that directly classifies images
    without contrastive learning. Requires a classification head.

    Note: This needs a linear classification layer on top of the encoder.
    """

    def __init__(
        self,
        num_classes: int = 5,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            num_classes: Number of classes to predict (5 for LC25000)
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute cross-entropy loss.

        Args:
            logits: [B, num_classes] raw logits from classification head
            labels: [B, 3] hierarchical labels - uses L1 (subtype) as target

        Returns:
            Dictionary with loss components
        """
        # Extract L1 labels (subtype level) - only use first B samples (not both views)
        batch_size = logits.shape[0]
        target_labels = labels[:batch_size, 1]  # [B]

        # Compute cross-entropy
        loss = self.criterion(logits, target_labels)

        # Compute accuracy for monitoring
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == target_labels).float().mean()

        return {
            'loss': loss,
            'ce_loss': loss.item(),
            'accuracy': accuracy.item()
        }


class CrossEntropyClassificationHead(nn.Module):
    """
    Classification head for Cross-Entropy baseline.

    This is used during pretraining for the CrossEntropy loss.
    For fair comparison, we'll still use linear probe evaluation after pretraining.
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 5):
        """
        Args:
            input_dim: Input feature dimension from backbone
            num_classes: Number of output classes
        """
        super(CrossEntropyClassificationHead, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, input_dim] features from backbone

        Returns:
            [B, num_classes] logits
        """
        return self.classifier(x)


def test_losses():
    """Test all baseline losses."""
    print("Testing Baseline Loss Functions\n" + "="*70)

    batch_size = 8
    feature_dim = 128

    # Create dummy features (normalized)
    features = torch.randn(2 * batch_size, feature_dim)
    features = F.normalize(features, dim=1)

    # Create hierarchical labels
    labels = torch.tensor([
        [0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3],
        [1, 2, 4], [1, 2, 5], [1, 3, 6], [1, 4, 7],
        # Repeat for second view
        [0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3],
        [1, 2, 4], [1, 2, 5], [1, 3, 6], [1, 4, 7],
    ])

    # Test MoCo
    print("\n1. Testing MoCoLoss...")
    moco_loss_fn = MoCoLoss(
        temperature=0.07, queue_size=256, feature_dim=feature_dim)
    moco_dict = moco_loss_fn(features, labels)
    print(f"   MoCo Loss: {moco_dict['moco_loss']:.4f}")
    assert 0 < moco_dict['loss'] < 10, "MoCo loss out of range"
    print("   [PASS] MoCo test passed")

    # Test SupCon
    print("\n2. Testing SupConLoss...")
    supcon_loss_fn = SupConLoss(temperature=0.07)
    supcon_dict = supcon_loss_fn(features, labels)
    print(f"   SupCon Loss: {supcon_dict['supcon_loss']:.4f}")
    assert 0 < supcon_dict['loss'] < 10, "SupCon loss out of range"
    print("   [PASS] SupCon test passed")

    # Test CrossEntropy
    print("\n3. Testing CrossEntropyLoss...")
    # Create logits (only for first batch, not both views)
    logits = torch.randn(batch_size, 5)
    ce_loss_fn = CrossEntropyLoss(num_classes=5)
    ce_dict = ce_loss_fn(logits, labels)
    print(f"   CE Loss: {ce_dict['ce_loss']:.4f}")
    print(f"   Accuracy: {ce_dict['accuracy']:.2%}")
    assert 0 < ce_dict['loss'] < 10, "CE loss out of range"
    print("   [PASS] CrossEntropy test passed")

    print("\n" + "="*70)
    print("All baseline loss tests passed! [SUCCESS]")


if __name__ == "__main__":
    test_losses()
