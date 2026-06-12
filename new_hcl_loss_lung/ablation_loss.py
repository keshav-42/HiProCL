"""
Ablation Loss Module for Component-wise Analysis

This module allows enabling/disabling individual loss components for ablation studies:
- L_instance: SimCLR-style instance discrimination
- L_subtype: Supervised contrastive at subtype level
- L_organ: Supervised contrastive at organ level
- L_prototypical: Prototypical loss with class centers

Based on ImprovedHierarchicalLoss but with component toggles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AblationLoss(nn.Module):
    """
    Ablation loss for testing individual and combined loss components.

    Enable/disable components via boolean flags.
    """

    def __init__(
        self,
        # Component toggles
        use_instance: bool = True,
        use_subtype: bool = True,
        use_organ: bool = True,
        use_prototypical: bool = True,
        # Loss weights
        alpha: float = 1.0,      # L_instance weight
        beta: float = 0.5,       # L_subtype weight
        gamma: float = 0.3,      # L_organ weight
        lambda_: float = 0.2,    # L_prototypical weight
        # Other parameters
        temperature: float = 0.5,
        num_subtypes: int = 5,
        num_organs: int = 2,
        feature_dim: int = 128,
        momentum: float = 0.99,
        eps: float = 1e-8
    ):
        """
        Args:
            use_instance: Enable L_instance component
            use_subtype: Enable L_subtype component
            use_organ: Enable L_organ component
            use_prototypical: Enable L_prototypical component
            alpha: Weight for L_instance
            beta: Weight for L_subtype
            gamma: Weight for L_organ
            lambda_: Weight for L_prototypical
            temperature: Temperature for contrastive losses
            num_subtypes: Number of subtype classes
            num_organs: Number of organ classes
            feature_dim: Feature dimension
            momentum: Momentum for prototype updates
            eps: Small epsilon for numerical stability
        """
        super(AblationLoss, self).__init__()

        # Component toggles
        self.use_instance = use_instance
        self.use_subtype = use_subtype
        self.use_organ = use_organ
        self.use_prototypical = use_prototypical

        # Validate at least one component is enabled
        if not any([use_instance, use_subtype, use_organ, use_prototypical]):
            raise ValueError("At least one loss component must be enabled!")

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.temperature = temperature
        self.momentum = momentum
        self.eps = eps

        # Initialize prototypes if using prototypical loss
        if self.use_prototypical:
            self.register_buffer('subtype_prototypes', torch.randn(num_subtypes, feature_dim))
            self.register_buffer('organ_prototypes', torch.randn(num_organs, feature_dim))
            self.subtype_prototypes = F.normalize(self.subtype_prototypes, dim=1)
            self.organ_prototypes = F.normalize(self.organ_prototypes, dim=1)

    @torch.no_grad()
    def _update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update prototypes with momentum."""
        if not self.use_prototypical:
            return

        l0_labels = labels[:, 0]
        l1_labels = labels[:, 1]

        # Update subtype prototypes
        for subtype_id in torch.unique(l1_labels):
            mask = (l1_labels == subtype_id)
            if mask.sum() > 0:
                current_center = features[mask].mean(dim=0)
                current_center = F.normalize(current_center, dim=0)
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
                self.organ_prototypes[organ_id] = (
                    self.momentum * self.organ_prototypes[organ_id] +
                    (1 - self.momentum) * current_center
                )
                self.organ_prototypes[organ_id] = F.normalize(
                    self.organ_prototypes[organ_id], dim=0
                )

    def _compute_instance_loss(self, features: torch.Tensor) -> torch.Tensor:
        """SimCLR-style instance discrimination."""
        batch_size = features.shape[0] // 2
        similarity = torch.matmul(features, features.T) / self.temperature

        # Positive pairs: (i, i+B) and (i+B, i)
        positive_mask = torch.zeros(2*batch_size, 2*batch_size, device=features.device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = 1
            positive_mask[i + batch_size, i] = 1

        self_mask = torch.eye(2*batch_size, device=features.device)
        similarity_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()
        exp_sim = torch.exp(similarity) * (1 - self_mask)
        denominator = exp_sim.sum(dim=1, keepdim=True) + self.eps
        log_prob = similarity - torch.log(denominator)
        loss = -(log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + self.eps)
        return loss.mean()

    def _compute_supervised_contrastive_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Supervised contrastive loss."""
        batch_size = features.shape[0]
        similarity = torch.matmul(features, features.T) / self.temperature
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        self_mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - self_mask)

        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        similarity_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()
        exp_sim = torch.exp(similarity)
        denominator = (exp_sim * (1 - self_mask)).sum(dim=1, keepdim=True) + self.eps
        log_prob = similarity - torch.log(denominator)
        num_positives = positive_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1.0)
        loss = -(log_prob * positive_mask).sum(dim=1) / num_positives
        mask_valid = (num_positives > 0).float()
        loss = (loss * mask_valid).sum() / (mask_valid.sum() + self.eps)
        return loss

    def _compute_prototypical_loss(
        self, features: torch.Tensor, labels: torch.Tensor, level: str = 'subtype'
    ) -> torch.Tensor:
        """Prototypical loss."""
        if level == 'subtype':
            prototypes = self.subtype_prototypes
        elif level == 'organ':
            prototypes = self.organ_prototypes
        else:
            raise ValueError(f"Unknown level: {level}")

        similarity = torch.matmul(features, prototypes.T) / self.temperature
        loss = F.cross_entropy(similarity, labels)
        return loss

    def _compute_prototype_separation_loss(self) -> torch.Tensor:
        """Encourage separation between prototypes (only for enabled hierarchies)."""
        total_sep_loss = 0.0

        if self.use_subtype:
            subtype_sim = torch.matmul(self.subtype_prototypes, self.subtype_prototypes.T)
            mask = 1 - torch.eye(subtype_sim.shape[0], device=subtype_sim.device)
            subtype_sep_loss = (subtype_sim * mask).sum() / (mask.sum() + self.eps)
            total_sep_loss += subtype_sep_loss

        if self.use_organ:
            organ_sim = torch.matmul(self.organ_prototypes, self.organ_prototypes.T)
            mask = 1 - torch.eye(organ_sim.shape[0], device=organ_sim.device)
            organ_sep_loss = (organ_sim * mask).sum() / (mask.sum() + self.eps)
            total_sep_loss += organ_sep_loss

        return total_sep_loss

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute loss with enabled components only.

        Returns dict with all component values (0.0 if disabled).
        """
        # Update prototypes if using prototypical loss
        self._update_prototypes(features, labels)

        # Extract labels
        l0_labels = labels[:, 0]  # Organ
        l1_labels = labels[:, 1]  # Subtype

        # Initialize all components
        l_instance = torch.tensor(0.0, device=features.device)
        l_subtype = torch.tensor(0.0, device=features.device)
        l_organ = torch.tensor(0.0, device=features.device)
        l_prototypical = torch.tensor(0.0, device=features.device)
        l_proto_subtype = torch.tensor(0.0, device=features.device)
        l_proto_organ = torch.tensor(0.0, device=features.device)
        l_proto_sep = torch.tensor(0.0, device=features.device)

        # Compute enabled components
        if self.use_instance:
            l_instance = self._compute_instance_loss(features)

        if self.use_subtype:
            l_subtype = self._compute_supervised_contrastive_loss(features, l1_labels)

        if self.use_organ:
            l_organ = self._compute_supervised_contrastive_loss(features, l0_labels)

        if self.use_prototypical:
            # Only compute prototypical loss for enabled hierarchical levels
            if self.use_subtype:
                l_proto_subtype = self._compute_prototypical_loss(features, l1_labels, 'subtype')
            if self.use_organ:
                l_proto_organ = self._compute_prototypical_loss(features, l0_labels, 'organ')

            # Prototype separation loss (only if at least one hierarchy is enabled)
            if self.use_subtype or self.use_organ:
                l_proto_sep = self._compute_prototype_separation_loss()

            l_prototypical = l_proto_subtype + l_proto_organ + 0.1 * l_proto_sep

        # Total weighted loss
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

    def get_config_name(self) -> str:
        """Get a descriptive name for this configuration."""
        components = []
        if self.use_instance:
            components.append('instance')
        if self.use_subtype:
            components.append('subtype')
        if self.use_organ:
            components.append('organ')
        if self.use_prototypical:
            components.append('proto')

        if len(components) == 1:
            return f"only_{components[0]}"
        elif len(components) == 4:
            return "full"
        else:
            return "_".join(components)
