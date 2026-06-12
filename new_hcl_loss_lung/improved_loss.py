"""
Improved Hierarchical Contrastive Loss - Fixed and Redesigned

This module implements a corrected hierarchical contrastive loss that addresses
the critical issues in the original design:

L_total = α·L_instance + β·L_subtype + γ·L_organ + λ·L_prototypical

Key Improvements:
1. Non-contradictory objectives
2. Proper InfoNCE formulation with correct negative sampling
3. Guaranteed positive pairs via instance-level contrast
4. Balanced loss magnitudes
5. Prototypical loss with momentum-updated centers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedHierarchicalLoss(nn.Module):
    """
    Improved Hierarchical Contrastive Loss for LC25000 dataset.

    Components:
    1. L_instance: SimCLR-style instance discrimination (two views)
    2. L_subtype: Supervised contrastive loss at subtype level
    3. L_organ: Supervised contrastive loss at organ level
    4. L_prototypical: Prototypical separation with class centers
    """

    def __init__(
        self,
        alpha: float = 1.0,      # L_instance weight
        beta: float = 0.5,       # L_subtype weight
        gamma: float = 0.3,      # L_organ weight
        lambda_: float = 0.2,    # L_prototypical weight
        temperature: float = 0.5,
        num_subtypes: int = 5,
        num_organs: int = 2,
        feature_dim: int = 128,
        momentum: float = 0.99,  # For prototype momentum update
        eps: float = 1e-8
    ):
        """
        Args:
            alpha: Weight for instance-level contrastive loss
            beta: Weight for subtype-level supervised contrastive loss
            gamma: Weight for organ-level supervised contrastive loss
            lambda_: Weight for prototypical loss
            temperature: Temperature for contrastive losses
            num_subtypes: Number of subtype classes (5 for LC25000)
            num_organs: Number of organ classes (2 for LC25000)
            feature_dim: Dimension of feature vectors
            momentum: Momentum for prototype updates
            eps: Small epsilon for numerical stability
        """
        super(ImprovedHierarchicalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.temperature = temperature
        self.momentum = momentum
        self.eps = eps

        # Initialize prototypes (class centers)
        self.register_buffer('subtype_prototypes', torch.randn(num_subtypes, feature_dim))
        self.register_buffer('organ_prototypes', torch.randn(num_organs, feature_dim))

        # Normalize prototypes
        self.subtype_prototypes = F.normalize(self.subtype_prototypes, dim=1)
        self.organ_prototypes = F.normalize(self.organ_prototypes, dim=1)

    @torch.no_grad()
    def _update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Update class prototypes with momentum.

        Args:
            features: [2B, D] normalized features
            labels: [2B, 3] hierarchical labels
        """
        l0_labels = labels[:, 0]  # Organ labels
        l1_labels = labels[:, 1]  # Subtype labels

        # Update subtype prototypes
        for subtype_id in torch.unique(l1_labels):
            mask = (l1_labels == subtype_id)
            if mask.sum() > 0:
                current_center = features[mask].mean(dim=0)
                current_center = F.normalize(current_center, dim=0)

                # Momentum update
                self.subtype_prototypes[subtype_id] = (
                    self.momentum * self.subtype_prototypes[subtype_id] +
                    (1 - self.momentum) * current_center
                )
                self.subtype_prototypes[subtype_id] = F.normalize(
                    self.subtype_prototypes[subtype_id], dim=0
                )

        # Update organ prototypes
        for organ_id in torch.unique(l0_labels):
            mask = (l0_labels == organ_id)
            if mask.sum() > 0:
                current_center = features[mask].mean(dim=0)
                current_center = F.normalize(current_center, dim=0)

                # Momentum update
                self.organ_prototypes[organ_id] = (
                    self.momentum * self.organ_prototypes[organ_id] +
                    (1 - self.momentum) * current_center
                )
                self.organ_prototypes[organ_id] = F.normalize(
                    self.organ_prototypes[organ_id], dim=0
                )

    def _compute_instance_loss(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SimCLR-style instance discrimination loss.

        This pulls together two augmented views of the same image.
        Guaranteed to have positive pairs!

        Args:
            features: [2B, D] where first B are view1, second B are view2

        Returns:
            Scalar loss value
        """
        batch_size = features.shape[0] // 2

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Create masks for positive pairs
        # Positive pairs: (i, i+B) and (i+B, i)
        positive_mask = torch.zeros(2*batch_size, 2*batch_size, device=features.device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = 1
            positive_mask[i + batch_size, i] = 1

        # Mask out self-similarity
        self_mask = torch.eye(2*batch_size, device=features.device)

        # For numerical stability
        similarity_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()

        # Compute exp(similarity)
        exp_sim = torch.exp(similarity) * (1 - self_mask)

        # Denominator: sum over all samples except self
        denominator = exp_sim.sum(dim=1, keepdim=True) + self.eps

        # Numerator: only positive pairs
        log_prob = similarity - torch.log(denominator)

        # Average over positive pairs
        loss = -(log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + self.eps)
        loss = loss.mean()

        return loss

    def _compute_supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Samples with same label are positives, others are negatives.

        Args:
            features: [2B, D] normalized features
            labels: [2B] class labels

        Returns:
            Scalar loss value
        """
        batch_size = features.shape[0]

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Create positive mask: same label
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()

        # Remove self-similarity
        self_mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - self_mask)

        # Check if there are any positive pairs
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # For numerical stability
        similarity_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()

        # Compute exp(similarity)
        exp_sim = torch.exp(similarity)

        # Denominator: sum over all samples except self
        denominator = (exp_sim * (1 - self_mask)).sum(dim=1, keepdim=True) + self.eps

        # Log probability
        log_prob = similarity - torch.log(denominator)

        # Average over positive pairs
        num_positives = positive_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1.0)

        loss = -(log_prob * positive_mask).sum(dim=1) / num_positives

        # Only compute loss for samples that have positive pairs
        mask_valid = (num_positives > 0).float()
        loss = (loss * mask_valid).sum() / (mask_valid.sum() + self.eps)

        return loss

    def _compute_prototypical_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        level: str = 'subtype'
    ) -> torch.Tensor:
        """
        Compute prototypical loss: push features towards their class prototype
        and away from other class prototypes.

        Args:
            features: [2B, D] normalized features
            labels: [2B] class labels at this level
            level: 'subtype' or 'organ'

        Returns:
            Scalar loss value
        """
        if level == 'subtype':
            prototypes = self.subtype_prototypes
        elif level == 'organ':
            prototypes = self.organ_prototypes
        else:
            raise ValueError(f"Unknown level: {level}")

        # Compute distances to all prototypes
        # similarity: [2B, num_classes]
        similarity = torch.matmul(features, prototypes.T) / self.temperature

        # Cross-entropy loss (push towards correct prototype)
        loss = F.cross_entropy(similarity, labels)

        return loss

    def _compute_prototype_separation_loss(self) -> torch.Tensor:
        """
        Encourage separation between class prototypes.

        Returns:
            Scalar loss value
        """
        # Subtype prototype separation
        subtype_sim = torch.matmul(self.subtype_prototypes, self.subtype_prototypes.T)
        # Mask out diagonal
        mask = 1 - torch.eye(subtype_sim.shape[0], device=subtype_sim.device)
        subtype_sep_loss = (subtype_sim * mask).sum() / (mask.sum() + self.eps)

        # Organ prototype separation
        organ_sim = torch.matmul(self.organ_prototypes, self.organ_prototypes.T)
        mask = 1 - torch.eye(organ_sim.shape[0], device=organ_sim.device)
        organ_sep_loss = (organ_sim * mask).sum() / (mask.sum() + self.eps)

        # Want to minimize similarity (maximize separation)
        return subtype_sep_loss + organ_sep_loss

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Compute total hierarchical contrastive loss.

        Args:
            features: [2B, D] normalized features (two views concatenated)
            labels: [2B, 3] hierarchical labels (l0, l1, l2)

        Returns:
            Dictionary with loss components
        """
        # Update prototypes (no grad)
        self._update_prototypes(features, labels)

        # Extract label levels
        l0_labels = labels[:, 0]  # Organ
        l1_labels = labels[:, 1]  # Subtype

        # 1. Instance-level loss (SimCLR)
        l_instance = self._compute_instance_loss(features)

        # 2. Subtype-level supervised contrastive loss
        l_subtype = self._compute_supervised_contrastive_loss(features, l1_labels)

        # 3. Organ-level supervised contrastive loss
        l_organ = self._compute_supervised_contrastive_loss(features, l0_labels)

        # 4. Prototypical losses
        l_proto_subtype = self._compute_prototypical_loss(features, l1_labels, 'subtype')
        l_proto_organ = self._compute_prototypical_loss(features, l0_labels, 'organ')

        # 5. Prototype separation loss
        l_proto_sep = self._compute_prototype_separation_loss()

        # Combine prototypical components
        l_prototypical = l_proto_subtype + l_proto_organ + 0.1 * l_proto_sep

        # Total loss
        total_loss = (
            self.alpha * l_instance +
            self.beta * l_subtype +
            self.gamma * l_organ +
            self.lambda_ * l_prototypical
        )

        return {
            'loss': total_loss,
            'l_instance': l_instance.item(),
            'l_subtype': l_subtype.item(),
            'l_organ': l_organ.item(),
            'l_prototypical': l_prototypical.item(),
            'l_proto_subtype': l_proto_subtype.item(),
            'l_proto_organ': l_proto_organ.item(),
            'l_proto_sep': l_proto_sep.item()
        }


def test_improved_loss():
    """Test function to verify improved loss works correctly."""
    print("Testing ImprovedHierarchicalLoss...")

    batch_size = 4
    feature_dim = 128

    # Create dummy features (normalized)
    features = torch.randn(2 * batch_size, feature_dim)
    features = F.normalize(features, dim=1)

    # Create hierarchical labels
    labels = torch.tensor([
        [0, 0, 0],  # colon_aca, instance 0
        [0, 0, 1],  # colon_aca, instance 1
        [0, 1, 2],  # colon_benign, instance 2
        [1, 2, 3],  # lung_aca, instance 3
        # Repeat for second view
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 3],
    ])

    # Initialize loss
    loss_fn = ImprovedHierarchicalLoss(
        alpha=1.0, beta=0.5, gamma=0.3, lambda_=0.2,
        temperature=0.5,
        num_subtypes=5, num_organs=2, feature_dim=feature_dim
    )

    # Compute loss
    loss_dict = loss_fn(features, labels)

    print(f"\nLoss Components:")
    print(f"  Total Loss: {loss_dict['loss']:.4f}")
    print(f"  L_instance: {loss_dict['l_instance']:.4f}")
    print(f"  L_subtype: {loss_dict['l_subtype']:.4f}")
    print(f"  L_organ: {loss_dict['l_organ']:.4f}")
    print(f"  L_prototypical: {loss_dict['l_prototypical']:.4f}")
    print(f"    - L_proto_subtype: {loss_dict['l_proto_subtype']:.4f}")
    print(f"    - L_proto_organ: {loss_dict['l_proto_organ']:.4f}")
    print(f"    - L_proto_sep: {loss_dict['l_proto_sep']:.4f}")

    # Check reasonable ranges
    assert 0 < loss_dict['loss'] < 10, f"Total loss out of range: {loss_dict['loss']}"
    assert 0 <= loss_dict['l_instance'] < 5, f"Instance loss out of range: {loss_dict['l_instance']}"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_improved_loss()
