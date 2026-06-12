"""
Ablation Study Training Script for Loss Component Analysis

This script trains models with different loss component combinations
across multiple data percentages (2%, 5%, 10%, 20%, 40%).

Saves detailed metrics and checkpoints for each configuration.
"""

import os
import sys
import yaml
import argparse
import random
import csv
from pathlib import Path
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from new_hcl.dataset import LC25000Dataset
from new_hcl.augmentations import get_train_transforms_with_two_views
from new_hcl.models import HCLModel
from new_hcl.ablation_loss import AblationLoss

# Import shared functions from train.py
from train import (
    set_seed, split_dataset, get_pretrain_subset,
    collate_fn_two_views, validate
)


def train_one_epoch_ablation(
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
    csv_writer=None
):
    """Train for one epoch and log all metrics."""
    model.train()

    metrics_sum = {}
    num_batches = 0

    max_grad_norm = config.get('max_grad_norm', None)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP
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
            metrics_sum[key] += value if isinstance(value, float) else value.item()
        num_batches += 1

        # Update progress bar
        pbar_dict = {'loss': f"{loss.item():.4f}"}
        if 'l_instance' in loss_dict and loss_dict['l_instance'] > 0:
            pbar_dict['inst'] = f"{loss_dict['l_instance']:.4f}"
        if 'l_subtype' in loss_dict and loss_dict['l_subtype'] > 0:
            pbar_dict['subt'] = f"{loss_dict['l_subtype']:.4f}"
        if 'l_organ' in loss_dict and loss_dict['l_organ'] > 0:
            pbar_dict['orgn'] = f"{loss_dict['l_organ']:.4f}"
        if 'l_prototypical' in loss_dict and loss_dict['l_prototypical'] > 0:
            pbar_dict['prot'] = f"{loss_dict['l_prototypical']:.4f}"
        pbar.set_postfix(pbar_dict)

        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            for key, value in loss_dict.items():
                val = value if isinstance(value, float) else value.item()
                writer.add_scalar(f'Train/{key}', val, global_step)

    # Compute averages
    avg_metrics = {key: val / num_batches for key, val in metrics_sum.items()}

    # Write to CSV
    if csv_writer is not None:
        row = {'epoch': epoch, 'split': 'train'}
        row.update(avg_metrics)
        csv_writer.writerow(row)

    return avg_metrics


def main_ablation(config_path: str, pretrain_percentage: float, output_dir: str, seed: int = 42):
    """Main training function for ablation study."""

    # Set seed
    set_seed(seed)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create descriptive output directory
    config_name = Path(config_path).stem  # e.g., "component_only_instance"
    output_dir = Path(output_dir) / config_name / f"{int(pretrain_percentage*100)}pct"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()

    print(f"\n{'='*70}")
    print(f"Ablation Study: {config_name} - {pretrain_percentage*100:.0f}% Data")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load dataset
    print("Loading dataset...")
    full_dataset = LC25000Dataset(root_dir=config['data_path'], transform=None)

    # Split dataset
    train_indices, val_indices, test_indices = split_dataset(
        full_dataset,
        split_ratio=[0.8, 0.1, 0.1],
        seed=seed
    )

    # Get pretrain subset
    pretrain_indices = get_pretrain_subset(
        train_indices.copy(),
        pretrain_percentage,
        seed=seed
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

    # Create datasets
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

    # Create ablation loss
    criterion = AblationLoss(
        use_instance=config.get('use_instance', True),
        use_subtype=config.get('use_subtype', True),
        use_organ=config.get('use_organ', True),
        use_prototypical=config.get('use_prototypical', True),
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

    print(f"Loss configuration: {criterion.get_config_name()}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Learning rate scheduler with optional warmup
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.get('warmup_lr', 0.0001) / config['learning_rate'],
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
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )

    # Create GradScaler for AMP
    scaler = GradScaler(enabled=use_amp)

    # Setup CSV logging
    metrics_csv_path = output_dir / "metrics.csv"
    csv_file = open(metrics_csv_path, 'w', newline='')
    fieldnames = ['epoch', 'split', 'loss', 'l_instance', 'l_subtype', 'l_organ',
                  'l_prototypical', 'l_proto_subtype', 'l_proto_organ', 'l_proto_sep']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Training loop
    best_val_loss = float('inf')

    print("\nStarting training...")
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*70}")

        # Train
        train_metrics = train_one_epoch_ablation(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config, use_amp, writer, csv_writer
        )

        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        for key, value in train_metrics.items():
            if key != 'loss' and value > 0:
                print(f"  {key}: {value:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        print(f"\nValidation metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        for key, value in val_metrics.items():
            if key != 'loss' and value > 0:
                print(f"  {key}: {value:.4f}")

        # Write validation to CSV
        row = {'epoch': epoch, 'split': 'val'}
        row.update(val_metrics)
        csv_writer.writerow(row)
        csv_file.flush()

        # Log to tensorboard
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

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

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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

    csv_file.close()
    writer.close()

    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ablation study for loss components")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to component config file'
    )
    parser.add_argument(
        '--pretrain_percentage',
        type=float,
        required=True,
        help='Percentage of training data (e.g., 0.02 for 2%%)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_results',
        help='Output directory for ablation results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    main_ablation(args.config, args.pretrain_percentage, args.output_dir, args.seed)
