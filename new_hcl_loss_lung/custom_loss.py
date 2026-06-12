"""
Custom Hierarchical Contrastive Loss with careful masking logic.

This module implements the novel hierarchical contrastive loss:
L_total = α · L_organ + β · L_subtype + λ · L_center_dist

CRITICAL: Proper masking is essential for convergence!
- L_subtype: Same l1 (subtype) BUT different l2 (instance)
- L_organ: Same l0 (organ) BUT different l1 (subtype)
- Exclude self-comparisons in all cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomHCLLoss(nn.Module):
    """
    Custom Hierarchical Contrastive Loss with three components:
    1. L_organ: Separates different organs (coarse-grained)
    2. L_subtype: Separates different subtypes within same organ (fine-grained)
    3. L_center_dist: Maximizes distance between organ cluster centers
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_: float = 0.1,
        temperature: float = 0.07,
        eps: float = 1e-8
    ):
        """
        Args:
            alpha: Weight for L_organ (coarse-grained contrastive loss)
            beta: Weight for L_subtype (fine-grained contrastive loss)
            lambda_: Weight for L_center_dist (cluster center separation)
            temperature: Temperature parameter for NT-Xent loss
            eps: Small epsilon for numerical stability
        """
        super(CustomHCLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.temperature = temperature
        self.eps = eps

    def _compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix.

        Args:
            features: [2B, D] normalized feature vectors

        Returns:
            Similarity matrix [2B, 2B]
        """
        # Features are already L2-normalized from projection head
        # Compute cosine similarity: sim(i,j) = z_i · z_j^T
        similarity = torch.matmul(features, features.T)
        return similarity

    def _compute_nt_xent_loss(
        self,
        similarity: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            similarity: [2B, 2B] similarity matrix
            positive_mask: [2B, 2B] binary mask for positive pairs
            negative_mask: [2B, 2B] binary mask for negative pairs

        Returns:
            Scalar loss value
        """
        batch_size = similarity.shape[0]

        # Apply temperature scaling
        similarity = similarity / self.temperature

        # For numerical stability, subtract max
        similarity = similarity - torch.max(similarity, dim=1, keepdim=True)[0].detach()

        # Compute exp of similarities
        exp_sim = torch.exp(similarity)

        # Mask out invalid negatives (keep only true negatives)
        exp_sim_neg = exp_sim * negative_mask

        # Denominator: sum over all negatives
        denominator = exp_sim_neg.sum(dim=1, keepdim=True) + self.eps

        # Numerator: positive pairs
        numerator = exp_sim * positive_mask

        # Compute log probability
        log_prob = torch.log(numerator / denominator + self.eps)

        # Average over positive pairs (only where positive_mask is 1)
        num_positives = positive_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1.0)  # Avoid division by zero

        loss = -(log_prob * positive_mask).sum(dim=1) / num_positives
        loss = loss.mean()

        return loss

    def _create_masks(
        self,
        labels: torch.Tensor,
        level_a: int,
        level_b: int,
        same_level_a: bool = True,
        same_level_b: bool = False
    ) -> tuple:
        """
        Create positive and negative masks based on hierarchical labels.

        Args:
            labels: [2B, 3] tensor with (l0, l1, l2) labels
            level_a: First level to compare (0, 1, or 2)
            level_b: Second level to compare (0, 1, or 2)
            same_level_a: If True, positives must have same label at level_a
            same_level_b: If True, positives must have same label at level_b

        Returns:
            Tuple of (positive_mask, negative_mask)
        """
        batch_size = labels.shape[0]

        # Extract labels for the specified levels
        labels_a = labels[:, level_a].unsqueeze(1)  # [2B, 1]
        labels_b = labels[:, level_b].unsqueeze(1)  # [2B, 1]

        # Create pairwise comparison matrices
        same_a = (labels_a == labels_a.T).float()  # [2B, 2B]
        same_b = (labels_b == labels_b.T).float()  # [2B, 2B]
        diff_b = (labels_b != labels_b.T).float()  # [2B, 2B]

        # Create self-mask (exclude diagonal)
        self_mask = (1.0 - torch.eye(batch_size, device=labels.device))  # [2B, 2B]

        # Build positive mask based on conditions
        if same_level_a and same_level_b:
            # Both must be same: same_a AND same_b AND not_self
            positive_mask = same_a * same_b * self_mask
        elif same_level_a and not same_level_b:
            # Same A but different B: same_a AND diff_b AND not_self
            positive_mask = same_a * diff_b * self_mask
        else:
            raise ValueError(f"Unsupported mask configuration")

        # Negative mask: exclude self and exclude positives
        negative_mask = self_mask * (1.0 - positive_mask)

        return positive_mask, negative_mask

    def _compute_center_distance_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute center distance loss to maximize separation between organ clusters.

        Args:
            features: [2B, D] normalized feature vectors
            labels: [2B, 3] hierarchical labels

        Returns:
            Scalar loss value (negative distance to maximize)
        """
        l0_labels = labels[:, 0]  # Organ labels

        # Get unique organs
        unique_organs = torch.unique(l0_labels)

        if len(unique_organs) < 2:
            # If only one organ in batch, return zero loss
            return torch.tensor(0.0, device=features.device)

        # Compute center for each organ
        centers = []
        for organ_id in unique_organs:
            mask = (l0_labels == organ_id)
            organ_features = features[mask]
            center = organ_features.mean(dim=0)
            centers.append(center)

        # Compute pairwise distances between all centers
        centers = torch.stack(centers)  # [num_organs, D]

        # Compute Euclidean distance matrix
        distances = torch.cdist(centers, centers, p=2)  # [num_organs, num_organs]

        # Get upper triangular part (exclude diagonal)
        num_centers = centers.shape[0]
        triu_indices = torch.triu_indices(num_centers, num_centers, offset=1)
        pairwise_distances = distances[triu_indices[0], triu_indices[1]]

        # We want to MAXIMIZE distance, so minimize negative distance
        avg_distance = pairwise_distances.mean()
        loss = -avg_distance

        return loss

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Compute the total hierarchical contrastive loss.

        Args:
            features: [2B, D] normalized projected features from two views
            labels: [2B, 3] hierarchical labels (l0, l1, l2)

        Returns:
            Dictionary containing:
                - 'loss': Total weighted loss
                - 'l_organ': Organ-level contrastive loss
                - 'l_subtype': Subtype-level contrastive loss
                - 'l_center_dist': Center distance loss
        """
        # Compute similarity matrix
        similarity = self._compute_similarity_matrix(features)

        # --- L_organ: Same l0 (organ) BUT different l1 (subtype) ---
        # This separates different subtypes within the same organ
        positive_mask_organ, negative_mask_organ = self._create_masks(
            labels, level_a=0, level_b=1, same_level_a=True, same_level_b=False
        )

        l_organ = torch.tensor(0.0, device=features.device)
        if positive_mask_organ.sum() > 0:
            l_organ = self._compute_nt_xent_loss(
                similarity, positive_mask_organ, negative_mask_organ
            )

        # --- L_subtype: Same l1 (subtype) BUT different l2 (instance) ---
        # This groups instances of the same subtype together
        positive_mask_subtype, negative_mask_subtype = self._create_masks(
            labels, level_a=1, level_b=2, same_level_a=True, same_level_b=False
        )

        l_subtype = torch.tensor(0.0, device=features.device)
        if positive_mask_subtype.sum() > 0:
            l_subtype = self._compute_nt_xent_loss(
                similarity, positive_mask_subtype, negative_mask_subtype
            )

        # --- L_center_dist: Maximize distance between organ cluster centers ---
        l_center_dist = self._compute_center_distance_loss(features, labels)

        # Compute total weighted loss
        total_loss = (
            self.alpha * l_organ +
            self.beta * l_subtype +
            self.lambda_ * l_center_dist
        )

        return {
            'loss': total_loss,
            'l_organ': l_organ.item(),
            'l_subtype': l_subtype.item(),
            'l_center_dist': l_center_dist.item()
        }


def test_loss_masks():
    """
    Test function to verify mask creation is correct.
    """
    print("Testing CustomHCLLoss masking logic...")

    # Create dummy data
    batch_size = 4
    feature_dim = 128

    # Create features (random normalized vectors)
    features = torch.randn(2 * batch_size, feature_dim)
    features = F.normalize(features, dim=1)

    # Create labels with known structure
    # Format: (l0_organ, l1_subtype, l2_instance)
    labels = torch.tensor([
        [0, 0, 0],  # colon_aca, instance 0
        [0, 0, 1],  # colon_aca, instance 1
        [0, 1, 2],  # colon_benign, instance 2
        [1, 2, 3],  # lung_aca, instance 3
        [1, 2, 4],  # lung_aca, instance 4
        [1, 3, 5],  # lung_benign, instance 5
        [1, 4, 6],  # lung_scc, instance 6
        [1, 4, 7],  # lung_scc, instance 7
    ])

    # Initialize loss
    loss_fn = CustomHCLLoss(alpha=1.0, beta=1.0, lambda_=0.1, temperature=0.07)

    # Compute loss
    loss_dict = loss_fn(features, labels)

    print(f"\nLoss components:")
    print(f"  Total Loss: {loss_dict['loss']:.4f}")
    print(f"  L_organ: {loss_dict['l_organ']:.4f}")
    print(f"  L_subtype: {loss_dict['l_subtype']:.4f}")
    print(f"  L_center_dist: {loss_dict['l_center_dist']:.4f}")

    # Test mask creation
    print("\n--- Testing Mask Creation ---")

    # Test L_subtype mask (same l1, different l2)
    pos_mask, neg_mask = loss_fn._create_masks(
        labels, level_a=1, level_b=2, same_level_a=True, same_level_b=False
    )
    print(f"\nL_subtype positive pairs (same subtype, different instance):")
    print(f"  Pair (0,1): {pos_mask[0,1].item()} [Both colon_aca] - Expected: 1")
    print(f"  Pair (3,4): {pos_mask[3,4].item()} [Both lung_aca] - Expected: 1")
    print(f"  Pair (6,7): {pos_mask[6,7].item()} [Both lung_scc] - Expected: 1")
    print(f"  Pair (0,2): {pos_mask[0,2].item()} [Different subtype] - Expected: 0")
    print(f"  Number of positive pairs: {pos_mask.sum().item()}")

    # Test L_organ mask (same l0, different l1)
    pos_mask, neg_mask = loss_fn._create_masks(
        labels, level_a=0, level_b=1, same_level_a=True, same_level_b=False
    )
    print(f"\nL_organ positive pairs (same organ, different subtype):")
    print(f"  Pair (0,2): {pos_mask[0,2].item()} [Colon: aca vs benign] - Expected: 1")
    print(f"  Pair (3,5): {pos_mask[3,5].item()} [Lung: aca vs benign] - Expected: 1")
    print(f"  Pair (3,6): {pos_mask[3,6].item()} [Lung: aca vs scc] - Expected: 1")
    print(f"  Pair (0,3): {pos_mask[0,3].item()} [Different organ] - Expected: 0")
    print(f"  Number of positive pairs: {pos_mask.sum().item()}")

    print("\n✓ Masking test complete!")


if __name__ == "__main__":
    test_loss_masks()
