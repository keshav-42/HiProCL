"""
Linear probe evaluation script for pretrained HCL model.

This script:
1. Loads a pretrained backbone checkpoint
2. Freezes the backbone weights
3. Trains a linear classifier on top for l1 (subtype) classification
4. Reports accuracy, confusion matrix, and per-class metrics
"""

from new_hcl_loss_lung.models import HCLModel
from new_hcl_loss_lung.augmentations import get_eval_augmentation
from new_hcl_loss_lung.dataset import LC25000Dataset
import os
import sys
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))


class LinearClassifier(nn.Module):
    """Simple linear classifier for evaluation."""

    def __init__(self, input_dim: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def collate_fn_single_view(batch):
    """
    Collate function for single view (no augmentation pairs).

    Args:
        batch: List of (image, (l0, l1, l2))

    Returns:
        Tuple of (images, l1_labels)
    """
    images, labels = [], []

    for image, (l0, l1, l2) in batch:
        images.append(image)
        labels.append(l1)  # We classify l1 (subtype)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels


def train_linear_probe(
    backbone: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.01,
    use_amp: bool = True,
    save_dir: Path = None
):
    """
    Train linear classifier on frozen backbone features with AMP.

    Args:
        backbone: Frozen pretrained backbone
        classifier: Linear classifier to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        use_amp: Whether to use Automatic Mixed Precision
        save_dir: Directory to save best models (optional)

    Returns:
        Trained classifier
    """
    # Ensure backbone is frozen
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # Setup optimizer and loss
    optimizer = optim.SGD(classifier.parameters(), lr=lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_classifier_state = None
    best_epoch = 0

    # Create checkpoints directory if saving
    if save_dir is not None:
        checkpoint_dir = save_dir / "linear_probe_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Extract features (no gradients for backbone)
            with torch.no_grad(), autocast(enabled=use_amp):
                features = backbone(images)

            optimizer.zero_grad(set_to_none=True)

            # Forward through classifier with AMP
            with autocast(enabled=use_amp):
                logits = classifier(features)
                loss = criterion(logits, labels)

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validate
        val_acc, val_loss = evaluate_linear_probe(
            backbone, classifier, val_loader, device, use_amp
        )

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_classifier_state = classifier.state_dict().copy()
            best_epoch = epoch

            # Save checkpoint if directory provided
            if save_dir is not None:
                checkpoint_path = checkpoint_dir / \
                    f"best_classifier_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'train_loss': avg_train_loss
                }, checkpoint_path)
                print(f"  ✓ Saved best model to {checkpoint_path}")

        scheduler.step()

    # Load best model
    classifier.load_state_dict(best_classifier_state)
    print(f"\n{'='*70}")
    print(
        f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*70}")

    return classifier


@torch.no_grad()
def evaluate_linear_probe(
    backbone: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True
):
    """
    Evaluate linear classifier with AMP.

    Args:
        backbone: Frozen backbone
        classifier: Linear classifier
        dataloader: Dataloader to evaluate on
        device: Device to use
        use_amp: Whether to use Automatic Mixed Precision

    Returns:
        Tuple of (accuracy, loss)
    """
    backbone.eval()
    classifier.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with AMP
        with autocast(enabled=use_amp):
            features = backbone(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

        # Accumulate
        total_loss += loss.item()
        _, predicted = logits.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = 100. * accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    return accuracy, avg_loss


@torch.no_grad()
def get_predictions(
    backbone: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True
):
    """
    Get predictions for computing detailed metrics with AMP.

    Returns:
        Tuple of (all_labels, all_predictions)
    """
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Getting predictions"):
        images = images.to(device, non_blocking=True)

        # Extract features and classify with AMP
        with autocast(enabled=use_amp):
            features = backbone(images)
            logits = classifier(features)
            _, predicted = logits.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Path
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Path
):
    """Save detailed classification metrics."""
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)

    # Save to file
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LINEAR PROBE EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Macro F1 Score: {f1_macro:.4f}\n")
        f.write(f"Weighted F1 Score: {f1_weighted:.4f}\n\n")

        f.write("="*60 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n")

    print(f"Detailed metrics saved to {save_path}")

    # Also print to console
    print("\n" + "="*60)
    print("LINEAR PROBE EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print("\n" + report)


def main(args):
    """Main evaluation function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()  # Use AMP only on CUDA
    print(f"Using device: {device}")
    print(
        f"Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint.get('config', {})

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data split
    if args.data_split is not None:
        print(f"Loading data split from {args.data_split}...")
        split_data = np.load(args.data_split)
        train_indices = split_data['train_indices'].tolist()
        val_indices = split_data['val_indices'].tolist()
        test_indices = split_data['test_indices'].tolist()
    else:
        print("No data split provided, creating new split...")
        # Load full dataset and create split
        from train import split_dataset
        full_dataset = LC25000Dataset(root_dir=args.data_path, transform=None)
        train_indices, val_indices, test_indices = split_dataset(
            full_dataset, split_ratio=[0.8, 0.1, 0.1], seed=42
        )

    # Create datasets
    eval_transform = get_eval_augmentation(
        image_size=config.get('image_size', 224))

    print("Creating datasets...")
    full_dataset_temp = LC25000Dataset(root_dir=args.data_path, transform=None)

    train_dataset = LC25000Dataset(
        root_dir=args.data_path,
        transform=eval_transform,
        file_list=[full_dataset_temp.image_paths[i] for i in train_indices]
    )

    val_dataset = LC25000Dataset(
        root_dir=args.data_path,
        transform=eval_transform,
        file_list=[full_dataset_temp.image_paths[i] for i in val_indices]
    )

    test_dataset = LC25000Dataset(
        root_dir=args.data_path,
        transform=eval_transform,
        file_list=[full_dataset_temp.image_paths[i] for i in test_indices]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_single_view
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_single_view
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_single_view
    )

    # Load model
    print("Loading pretrained model...")
    model = HCLModel(
        pretrained_path=None,  # Already loaded from checkpoint
        backbone_dim=config.get('backbone_dim', 768),
        projection_hidden_dim=config.get('projection_hidden_dim', 2048),
        projection_output_dim=config.get('projection_dim', 128)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create linear classifier
    num_classes = 5  # LC25000 has 5 subtypes
    classifier = LinearClassifier(
        input_dim=config.get('backbone_dim', 768),
        num_classes=num_classes
    ).to(device)

    # Train linear probe
    print("\nTraining linear classifier...")
    classifier = train_linear_probe(
        backbone=model.backbone,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        use_amp=use_amp,
        save_dir=output_dir  # ✓ Now it will save checkpoints!
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc, test_loss = evaluate_linear_probe(
        model.backbone, classifier, test_loader, device, use_amp
    )

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Get predictions for detailed metrics
    y_true, y_pred = get_predictions(
        model.backbone, classifier, test_loader, device, use_amp
    )

    # Define class names
    class_names = ['colon_aca', 'colon_benign',
                   'lung_aca', 'lung_benign', 'lung_scc']

    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # Save detailed metrics
    metrics_path = output_dir / "evaluation_metrics.txt"
    save_metrics(y_true, y_pred, class_names, metrics_path)

    # Save trained classifier
    classifier_path = output_dir / "linear_classifier.pth"
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'num_classes': num_classes,
        'class_names': class_names
    }, classifier_path)
    print(f"\nLinear classifier saved to {classifier_path}")

    print("\n" + "="*60)
    print("LINEAR PROBE EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation for HCL model")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to pretrained model checkpoint'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to LC25000 dataset root directory'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default=None,
        help='Path to saved data split (.npz file from training)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='linear_probe_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train linear classifier'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate for linear classifier'
    )

    args = parser.parse_args()
    main(args)
