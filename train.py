"""
Training script for Hierarchical Contrastive Learning on LC25000 dataset.

This script:
1. Splits data into 80/10/10 (train/val/test)
2. Supports pretraining with different percentages of training data (1%, 5%, 10%, 20%, 40%)
3. Implements the custom hierarchical contrastive loss
4. Saves checkpoints and logs training progress
"""

from new_hcl_loss_lung.improved_loss import ImprovedHierarchicalLoss
from new_hcl_loss_lung.custom_loss import CustomHCLLoss
from new_hcl_loss_lung.models import HCLModel
from new_hcl_loss_lung.augmentations import get_train_transforms_with_two_views
from new_hcl_loss_lung.dataset import LC25000Dataset
import os
import sys
import yaml
import argparse
import random
from pathlib import Path
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset: LC25000Dataset, split_ratio: List[float] = [0.8, 0.1, 0.1], seed: int = 42):
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: The full dataset
        split_ratio: List of [train, val, test] ratios
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"

    # Get all indices
    num_samples = len(dataset)
    indices = list(range(num_samples))

    # Shuffle with seed
    random.Random(seed).shuffle(indices)

    # Calculate split points
    train_size = int(split_ratio[0] * num_samples)
    val_size = int(split_ratio[1] * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(
        f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    return train_indices, val_indices, test_indices


def get_pretrain_subset(train_indices: List[int], percentage: float, seed: int = 42):
    """
    Get a subset of training data for pretraining.

    Args:
        train_indices: Full list of training indices
        percentage: Percentage of data to use (e.g., 0.01 for 1%)
        seed: Random seed

    Returns:
        List of selected indices
    """
    num_samples = int(len(train_indices) * percentage)
    random.Random(seed).shuffle(train_indices)
    subset_indices = train_indices[:num_samples]

    print(
        f"Using {len(subset_indices)} samples ({percentage*100:.0f}% of training data)")

    return subset_indices


def collate_fn_two_views(batch):
    """
    Custom collate function for two-view data.

    Args:
        batch: List of tuples ((view1, view2), (l0, l1, l2))

    Returns:
        Tuple of (images, labels) where images = [2B, C, H, W] and labels = [2B, 3]
    """
    views1, views2, labels = [], [], []

    for (view1, view2), (l0, l1, l2) in batch:
        views1.append(view1)
        views2.append(view2)
        labels.append([l0, l1, l2])

    # Stack views: [B, C, H, W] each
    views1 = torch.stack(views1)
    views2 = torch.stack(views2)

    # Concatenate to create [2B, C, H, W]
    images = torch.cat([views1, views2], dim=0)

    # Create labels [2B, 3]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)

    return images, labels_tensor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: dict,
    use_amp: bool = True,
    writer: SummaryWriter = None,
    use_hcsc_full: bool = False
):
    """Train for one epoch with Automatic Mixed Precision."""
    model.train()

    # Initialize metric accumulators
    metrics_sum = {}
    num_batches = 0

    max_grad_norm = config.get('max_grad_norm', None)
    log_components = config.get('log_loss_components', False)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP
        if use_hcsc_full:
            # For HCSC full, split images into two views and process separately
            batch_size = images.shape[0] // 2
            images_q = images[:batch_size]
            images_k = images[batch_size:]
            
            with autocast(enabled=use_amp):
                features_q = model(images_q)
                features_k = model(images_k)
                loss_dict = criterion(features_q, features_k, labels[:batch_size])
                loss = loss_dict['loss']
        else:
            # Standard path: all views concatenated
            with autocast(enabled=use_amp):
                features = model(images)
                loss_dict = criterion(features, labels)
                loss = loss_dict['loss']

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping (optional)
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        for key, value in loss_dict.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0.0
            metrics_sum[key] += value if isinstance(
                value, float) else value.item()
        num_batches += 1

        # Update progress bar (show main components)
        pbar_dict = {'loss': f"{loss.item():.4f}"}
        if 'l_instance' in loss_dict:
            pbar_dict['inst'] = f"{loss_dict['l_instance']:.4f}"
        if 'l_subtype' in loss_dict:
            pbar_dict['subt'] = f"{loss_dict['l_subtype']:.4f}"
        if 'l_organ' in loss_dict:
            pbar_dict['orgn'] = f"{loss_dict['l_organ']:.4f}"
        if 'l_hproto' in loss_dict:
            pbar_dict['hpro'] = f"{loss_dict['l_hproto']:.4f}"
        pbar.set_postfix(pbar_dict)

        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            for key, value in loss_dict.items():
                val = value if isinstance(value, float) else value.item()
                writer.add_scalar(f'Train/{key}', val, global_step)

    # Compute averages
    avg_metrics = {key: val / num_batches for key, val in metrics_sum.items()}

    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    use_hcsc_full: bool = False
):
    """Validate the model with AMP."""
    model.eval()

    metrics_sum = {}
    num_batches = 0

    for images, labels in tqdm(dataloader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with AMP
        if use_hcsc_full:
            # For HCSC full, split images into two views and process separately
            batch_size = images.shape[0] // 2
            images_q = images[:batch_size]
            images_k = images[batch_size:]
            
            with autocast(enabled=use_amp):
                features_q = model(images_q)
                features_k = model(images_k)
                loss_dict = criterion(features_q, features_k, labels[:batch_size])
        else:
            # Standard path: all views concatenated
            with autocast(enabled=use_amp):
                features = model(images)
                loss_dict = criterion(features, labels)

        # Accumulate metrics
        for key, value in loss_dict.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0.0
            val = value.item() if torch.is_tensor(value) else value
            metrics_sum[key] += val
        num_batches += 1

    # Compute averages
    avg_metrics = {key: val / num_batches for key, val in metrics_sum.items()}

    return avg_metrics


def main(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir) / \
        f"pretrain_{int(args.pretrain_percentage*100)}pct"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()  # Use AMP only on CUDA
    print(f"Using device: {device}")
    print(
        f"Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  pretrain_percentage: {args.pretrain_percentage}")
    print(f"  use_amp: {use_amp}")
    print("="*50 + "\n")

    # Create full dataset (without augmentation for splitting)
    print("Loading dataset...")
    full_dataset = LC25000Dataset(root_dir=config['data_path'], transform=None)

    # Split dataset
    train_indices, val_indices, test_indices = split_dataset(
        full_dataset,
        split_ratio=[0.8, 0.1, 0.1],
        seed=args.seed
    )

    # Get pretrain subset
    pretrain_indices = get_pretrain_subset(
        train_indices.copy(),
        args.pretrain_percentage,
        seed=args.seed
    )

    # Save split indices
    split_save_path = output_dir / "data_split.npz"
    np.savez(
        split_save_path,
        train_indices=train_indices,
        pretrain_indices=pretrain_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )
    print(f"Saved data split to {split_save_path}")

    # Create datasets with augmentation
    train_transform = get_train_transforms_with_two_views(
        image_size=config.get('image_size', 224),
        s=config.get('color_jitter_strength', 1.0)
    )

    train_dataset = LC25000Dataset(
        root_dir=config['data_path'],
        transform=train_transform,
        file_list=[full_dataset.image_paths[i] for i in pretrain_indices]
    )

    val_dataset = LC25000Dataset(
        root_dir=config['data_path'],
        transform=train_transform,
        file_list=[full_dataset.image_paths[i] for i in val_indices]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn_two_views,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn_two_views,
        drop_last=False
    )

    # Create model
    print("Initializing model...")
    model = HCLModel(
        pretrained_path=config.get('pretrained_path', None),
        backbone_dim=config.get('backbone_dim', 768),
        projection_hidden_dim=config.get('projection_hidden_dim', 2048),
        projection_output_dim=config.get('projection_dim', 128)
    ).to(device)

    # Create loss function based on config
    loss_type = config.get('loss_type', 'original')
    print(f"Using loss function: {loss_type}")

    if loss_type == 'hcsc_full':
        # Full HCSC with momentum encoder + feature queue
        from new_hcl_loss_lung.hcsc_full_loss import HCSCFullLoss

        clusters_cfg = config.get('hcsc_num_clusters', [32, 16, 8])
        if isinstance(clusters_cfg, str):
            num_clusters = [int(x.strip()) for x in clusters_cfg.split(',') if x.strip()]
        else:
            num_clusters = [int(x) for x in clusters_cfg]

        criterion = HCSCFullLoss(
            base_model=model,
            dim=config.get('projection_dim', 128),
            queue_size=config.get('queue_size', 16384),
            momentum=config.get('momentum_encoder', 0.999),
            temperature=config.get('temperature', 0.2),
            num_clusters=num_clusters,
            kmeans_iters=config.get('hcsc_kmeans_iters', 10),
            instance_selection=config.get('hcsc_instance_selection', True),
            proto_selection=config.get('hcsc_proto_selection', True)
        ).to(device)

        # Flag that we're using dual encoders for training loop
        use_hcsc_full = True

    elif loss_type == 'hcsc':
        # Import locally so existing paths remain unchanged for other loss types.
        from new_hcl_loss_lung.hcsc_loss import HCSCHierarchicalKMeansLoss

        clusters_cfg = config.get('hcsc_num_clusters', [32, 16, 8])
        if isinstance(clusters_cfg, str):
            num_clusters = [int(x.strip()) for x in clusters_cfg.split(',') if x.strip()]
        else:
            num_clusters = [int(x) for x in clusters_cfg]

        criterion = HCSCHierarchicalKMeansLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 1.0),
            temperature=config.get('temperature', 0.2),
            num_clusters=num_clusters,
            kmeans_iters=config.get('hcsc_kmeans_iters', 10),
            kmeans_interval=config.get('hcsc_kmeans_interval', 1)
        ).to(device)

        use_hcsc_full = False

    elif loss_type == 'improved':
        criterion = ImprovedHierarchicalLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.5),
            gamma=config.get('gamma', 0.3),
            lambda_=config.get('lambda', 0.2),
            temperature=config.get('temperature', 0.5),
            num_subtypes=config.get('num_subtypes', 5),
            num_organs=config.get('num_organs', 2),
            feature_dim=config.get('projection_dim', 128),
            momentum=config.get('prototype_momentum', 0.99)
        ).to(device)
        use_hcsc_full = False

    else:  # original
        criterion = CustomHCLLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 1.0),
            lambda_=config.get('lambda', 0.1),
            temperature=config.get('temperature', 0.07)
        )
        use_hcsc_full = False

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Learning rate scheduler with optional warmup
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        # Create warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.get('warmup_lr', 0.0001) /
            config['learning_rate'],
            total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'] - warmup_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"Using warmup for {warmup_epochs} epochs")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )

    # Create GradScaler for AMP
    scaler = GradScaler(enabled=use_amp)

    # Training loop state (supports resume)
    start_epoch = 1
    best_val_loss = float('inf')

    # Optional resume from checkpoint
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Loading resume checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Continue from next epoch
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1

        # Restore best val if available
        if 'val_metrics' in checkpoint and isinstance(checkpoint['val_metrics'], dict):
            best_val_loss = float(checkpoint['val_metrics'].get('loss', best_val_loss))

        print(f"Resumed at epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    print("\nStarting training...")
    for epoch in range(start_epoch, config['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config, use_amp, writer, use_hcsc_full
        )

        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        # Print all available metrics
        for key, value in train_metrics.items():
            if key != 'loss':
                print(f"  {key}: {value:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp, use_hcsc_full)

        print(f"\nValidation metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        for key, value in val_metrics.items():
            if key != 'loss':
                print(f"  {key}: {value:.4f}")

        # Log to tensorboard
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if epoch % config.get('save_every', 10) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            print(f"\n✓ New best model saved! Val Loss: {best_val_loss:.4f}")

    # Save final model
    final_model_path = checkpoint_dir / "final_model.pth"
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)

    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {checkpoint_dir}")
    print(f"{'='*50}\n")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HCL model on LC25000 dataset")

    parser.add_argument(
        '--config',
        type=str,
        default='configs/hcl_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--pretrain_percentage',
        type=float,
        nargs='+',  # Accept one or more values
        default=[1.0],
        help='Percentage(s) of training data to use (e.g., 0.10 0.20 0.40 for 10%%, 20%%, 40%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    # Validate percentages
    valid_percentages = [0.01, 0.05, 0.10, 0.20, 0.40, 1.0]
    for pct in args.pretrain_percentage:
        if pct not in valid_percentages:
            print(
                f"Warning: {pct} is not a standard percentage. Valid: {valid_percentages}")

    # Run training for each percentage
    for pct in args.pretrain_percentage:
        print(f"\n{'='*60}")
        print(f"Starting training with {pct*100}% of training data")
        print(f"{'='*60}\n")

        # Create a modified args with single percentage
        args_single = argparse.Namespace(**vars(args))
        args_single.pretrain_percentage = pct

        main(args_single)
