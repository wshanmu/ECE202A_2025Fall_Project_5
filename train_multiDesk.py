"""
Training and Evaluation Pipeline for SpatioTemporal ResNet HR Estimation

This script provides a complete training pipeline with:
- Hydra configuration management
- Weights & Biases logging
- Early stopping and checkpointing
- Gradient clipping
- Mixed precision training (optional)
- Reproducible training with seed control

Usage:
    python train.py                          # Use default config
    python train.py optimizer=sgd            # Override optimizer
    python train.py train.max_epochs=200     # Override specific param
    python train.py --multirun seed=1,2,3    # Multiple runs with different seeds
"""

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import profiler as torch_profiler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.multiDeskDataset import create_dataloaders, create_full_range_dataloaders
from src.models.antennaHybridNet import AntennaHybridNet


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value

    Note:
        Setting deterministic mode may impact performance.
        See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Enable deterministic mode (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def reshape_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Reshape batch from dataset format to model input format.

    Dataset output: (B, A2, C2, T, R)
        B = batch size
        A2 = 2 (two antennas/pairs)
        C2 = 2 (magnitude + phase)
        T = 1536 (time frames)
        R = 896 (range bins)

    Model input: (B, T, S)
        B = batch size
        T = 1536 (time frames)
        S = A2 * C2 * R (spatial dimension)

    Args:
        x: Input tensor of shape (B, A2, C2, T, R)

    Returns:
        Reshaped tensor of shape (B, T, S)
    """
    B, A2, C2, T, R = x.shape

    # Permute to move time to second axis: (B, T, A2, C2, R)
    x = x.permute(0, 3, 1, 2, 4)

    # Flatten spatial dimensions: (B, T, A2*C2*R)
    x = x.reshape(B, T, A2 * C2 * R)

    return x


def get_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: PyTorch model
        cfg: Optimizer configuration

    Returns:
        Configured optimizer
    """
    params = model.parameters()

    if cfg.name.lower() == "adam":
        return optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    elif cfg.name.lower() == "sgd":
        return optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov
        )
    elif cfg.name.lower() == "adamw":
        return optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=getattr(cfg, 'betas', [0.9, 0.999]),
            eps=getattr(cfg, 'eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.name}")


def get_scheduler(optimizer: optim.Optimizer, cfg: DictConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        cfg: Scheduler configuration

    Returns:
        Configured scheduler or None
    """
    if cfg.name.lower() == "none":
        return None
    elif cfg.name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.eta_min
        )
    elif cfg.name.lower() == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma
        )
    elif cfg.name.lower() == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=getattr(cfg, 'mode', 'min'),
            factor=getattr(cfg, 'factor', 0.1),
            patience=getattr(cfg, 'patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.name}")


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_clip_val: Optional[float] = None,
    use_amp: bool = False,
    profiler: Optional[Any] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: Gradient scaler for mixed precision
        gradient_clip_val: Max gradient norm for clipping
        use_amp: Use automatic mixed precision
        profiler: Optional torch.profiler object used for tracing

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_data, batch_labels in pbar:
        with torch_profiler.record_function("train.to_device"):
            batch_data = batch_data.to(device)
            if batch_labels.dim() == 1:
                batch_labels = batch_labels.to(device).float().unsqueeze(1)  # (B,) -> (B, 1)
            else:
                batch_labels = batch_labels.to(device).float()
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if use_amp:
            with torch_profiler.record_function("train.forward"):
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels.float())
                all_preds.append(outputs.cpu())
                all_labels.append(batch_labels.cpu())

            # Backward pass with gradient scaling
            with torch_profiler.record_function("train.backward"):
                scaler.scale(loss).backward()

            # Gradient clipping
            if gradient_clip_val is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            # Optimizer step
            with torch_profiler.record_function("train.optimizer_step"):
                scaler.step(optimizer)
                scaler.update()
        else:
            # Standard forward pass
            with torch_profiler.record_function("train.forward"):
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels.float())
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())

            # Backward pass
            with torch_profiler.record_function("train.backward"):
                loss.backward()

            # Gradient clipping
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            # Optimizer step
            with torch_profiler.record_function("train.optimizer_step"):
                optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if profiler is not None:
            profiler.step()
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # Calculate accuracy
    predictions_binary = (all_preds > 0.6).float()
    accuracy = (predictions_binary == all_labels).float().mean().item()

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy
    }

    return metrics


from sklearn.metrics import precision_recall_fscore_support

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for batch_data, batch_labels in pbar:
        # Move to device
        batch_data = batch_data.to(device)
        if batch_labels.dim() == 1:
            batch_labels = batch_labels.to(device).float().unsqueeze(1)  # (B,) -> (B, 1)
        else:
            batch_labels = batch_labels.to(device).float()

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels.float())

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Store predictions and labels for metrics
        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu())
        all_labels.append(batch_labels.cpu())

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # print(all_preds.cpu().numpy(), all_labels.cpu().numpy())

    # Calculate metrics
    # For binary classification, we could compute accuracy
    # Assuming threshold of 0.6 for binary classification
    predictions_binary = (all_preds > 0.6).float()
    accuracy = (predictions_binary == all_labels).float().mean().item()

    # calculate the precision, recall, f1-score

    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     all_labels.cpu().numpy(), predictions_binary.cpu().numpy(), average='binary'
    # )

    # Calculate MAE (Mean Absolute Error) for regression-like metric
    mae = (all_preds - all_labels).abs().mean().item()

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy,
        'mae': mae,
        # 'precision': precision,
        # 'recall': recall,
        # 'f1_score': f1,
    }

    return metrics


@torch.no_grad()
def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Test the model (same as validation but with different naming).

    Args:
        model: PyTorch model
        dataloader: Test data loader
        criterion: Loss function
        device: Device to test on

    Returns:
        Dictionary with test metrics
    """
    return validate_epoch(model, dataloader, criterion, device)


# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """
    Manages model checkpoints during training.

    Saves top-k checkpoints based on a metric and optionally the last checkpoint.
    """

    def __init__(
        self,
        save_dir: Path,
        save_top_k: int = 1,
        save_last: bool = True,
        metric_name: str = "val/loss",
        mode: str = "min"
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save the last checkpoint
            metric_name: Metric to monitor for best checkpoint
            mode: "min" or "max" for best metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_top_k = save_top_k
        self.save_last = save_last
        self.metric_name = metric_name
        self.mode = mode

        self.best_metrics = []  # List of (metric_value, epoch, checkpoint_path)

        # Determine comparison function
        self.is_better = (lambda new, old: new < old) if mode == "min" else (lambda new, old: new > old)

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        metric_value: float,
        is_last: bool = False
    ):
        """
        Save a checkpoint if it's among the top-k or if it's the last epoch.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            metric_value: Value of the monitored metric
            is_last: Whether this is the last epoch
        """
        # Save last checkpoint if requested
        if is_last and self.save_last:
            last_path = self.save_dir / "last.ckpt"
            self._save_checkpoint_file(last_path, epoch, model, optimizer, scheduler, metric_value)
            print(f"  Saved last checkpoint: {last_path}")

        # Check if this is a top-k checkpoint
        should_save = False

        if len(self.best_metrics) < self.save_top_k:
            should_save = True
        else:
            # Find worst checkpoint among saved
            worst_metric = max(self.best_metrics, key=lambda x: x[0]) if self.mode == "min" else min(self.best_metrics, key=lambda x: x[0])

            if self.is_better(metric_value, worst_metric[0]):
                should_save = True

                # Remove worst checkpoint
                worst_checkpoint_path = worst_metric[2]
                if worst_checkpoint_path.exists():
                    worst_checkpoint_path.unlink()

                self.best_metrics.remove(worst_metric)

        if should_save:
            # Save checkpoint
            checkpoint_path = self.save_dir / f"epoch_{epoch:03d}_metric_{metric_value:.4f}.ckpt"
            self._save_checkpoint_file(checkpoint_path, epoch, model, optimizer, scheduler, metric_value)

            self.best_metrics.append((metric_value, epoch, checkpoint_path))
            print(f"  Saved best checkpoint: {checkpoint_path}")

            # Also save as best.ckpt for easy loading
            best_path = self.save_dir / "best.ckpt"
            self._save_checkpoint_file(best_path, epoch, model, optimizer, scheduler, metric_value)

    def _save_checkpoint_file(
        self,
        path: Path,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        metric_value: float
    ):
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, path)

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_path = self.save_dir / "best.ckpt"
        return best_path if best_path.exists() else None


# ============================================================================
# Main Training Function
# ============================================================================

def train(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("="*80)
    print("Configuration:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Set random seed
    set_seed(cfg.seed)
    print(f"\nSet random seed: {cfg.seed}")

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            name=cfg.experiment_name,
            notes=cfg.wandb.notes,
            save_code=cfg.wandb.save_code
        )

    # Create data loaders
    print("\nCreating data loaders...")
    # Example usage
    data_directory = "./src/data/data/"
    label_yaml = "./src/data/dataset_labels_oneLongDesk_LOVertical_gated.yaml"
    # label_yaml = "./src/data/dataset_labels_oneLongDesk_LOHorizontal_gated.yaml"

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_directory,
        label_yaml_path=label_yaml,
        window_size=1536,
        stride=768,
        sender_idx=None,
        batch_size=16,
        num_workers=8,
        # Normalization
        use_minmax_norm=True,
        norm_range=(0, 1),
        # Augmentation (train only)
        use_gaussian_noise=True,
        noise_std=0.05,
        use_time_warp=True,
        time_warp_sigma=0.3,
        time_warp_knots=4,
        single_antenna=True,
    )


    # train_loader, val_loader, test_loader = create_full_range_dataloaders(
    #     data_dir=data_directory,
    #     label_yaml_path="./src/data/dataset_labels_oneLongDesk_LOVertical_fullRange.yaml",
    #     window_size=1536,
    #     stride=768,
    #     range_window=32,
    #     range_stride=32,
    #     batch_size=16,
    #     num_workers=8,
    #     norm_range=(0, 1),
    #     noise_std=0.05,
    #     use_minmax_norm=True,
    #     use_gaussian_noise=True,
    #     use_time_warp=True,
    #     time_warp_sigma=0.3,
    #     time_warp_knots=4,
    # )

    # get number of negative and positive samples in train_loader
    num_neg = 0
    num_pos = 0
    for _, labels in train_loader:
        num_neg += (labels == 0).sum().item()
        num_pos += (labels == 1).sum().item()
    print(f"  Training samples: {len(train_loader.dataset)} (Neg: {num_neg}, Pos: {num_pos})")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = AntennaHybridNet(num_classes=1)
    # apply gaussian initialization
    def gaussian_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(gaussian_init)
    # model = SpatioTemporalResNetHR_V2(
    #     spatial_handling="pool")
    
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create optimizer
    print(f"\nCreating optimizer: {cfg.optimizer.name}")
    optimizer = get_optimizer(model, cfg.optimizer)

    # Create scheduler
    print(f"Creating scheduler: {cfg.scheduler.name}")
    scheduler = get_scheduler(optimizer, cfg.scheduler)

    # Create loss function (Binary Cross Entropy for binary classification)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(num_neg / num_pos)]).to(device))

    # Create checkpoint manager
    checkpoint_dir = Path.cwd() / "checkpoints"
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_dir,
        save_top_k=cfg.train.save_top_k,
        save_last=cfg.train.save_last,
        metric_name=cfg.train.checkpoint_metric,
        mode=cfg.train.checkpoint_mode
    )

    # Setup optional profiler
    profiler_cfg = getattr(cfg.train, "profiling", None)
    profiler_enabled = bool(profiler_cfg and profiler_cfg.enabled)
    profiler_context = nullcontext()

    if profiler_enabled:
        log_dir = Path(profiler_cfg.log_dir) if profiler_cfg.log_dir else Path.cwd() / "tb_profiler"
        log_dir.mkdir(parents=True, exist_ok=True)

        activities = []
        for activity in profiler_cfg.activities:
            activity_upper = activity.upper()
            if activity_upper == "CPU":
                activities.append(torch_profiler.ProfilerActivity.CPU)
            elif activity_upper == "CUDA":
                if torch.cuda.is_available():
                    activities.append(torch_profiler.ProfilerActivity.CUDA)
            else:
                warnings.warn(f"Unknown profiler activity '{activity}'. Skipping.")

        if not activities:
            activities = [torch_profiler.ProfilerActivity.CPU]

        profiler_schedule = torch_profiler.schedule(
            wait=profiler_cfg.wait_steps,
            warmup=profiler_cfg.warmup_steps,
            active=profiler_cfg.active_steps,
            repeat=profiler_cfg.repeat,
            skip_first=profiler_cfg.skip_first
        )

        trace_handler = torch_profiler.tensorboard_trace_handler(str(log_dir))

        profiler_context = torch_profiler.profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=trace_handler,
            record_shapes=profiler_cfg.record_shapes,
            profile_memory=profiler_cfg.profile_memory,
            with_stack=profiler_cfg.with_stack,
            with_flops=getattr(profiler_cfg, "with_flops", False),
            with_modules=getattr(profiler_cfg, "with_modules", False)
        )

        print(f"\nTorch profiler enabled. Traces will be written to: {log_dir}")

    # Setup mixed precision training
    scaler = GradScaler() if cfg.train.use_amp else None

    # Early stopping tracking
    best_metric = float('inf') if cfg.train.early_stopping_mode == "min" else float('-inf')
    patience_counter = 0
    is_better_fn = (lambda new, best: new < best) if cfg.train.early_stopping_mode == "min" else (lambda new, best: new > best)

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    with profiler_context as profiler:
        for epoch in range(1, cfg.train.max_epochs + 1):
            print(f"\nEpoch {epoch}/{cfg.train.max_epochs}")
            print("-" * 40)

            # Train
            train_metrics = train_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                gradient_clip_val=cfg.train.gradient_clip_val,
                use_amp=cfg.train.use_amp,
                profiler=profiler if profiler_enabled else None
            )

            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Train accuracy: {train_metrics['accuracy']:.4f}")

            # Log to wandb
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

            # Validation
            if epoch % cfg.train.val_interval == 0:
                val_metrics = validate_epoch(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device
                )

                print(f"  Val loss: {val_metrics['loss']:.4f}")
                print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Val MAE: {val_metrics['mae']:.4f}")

                # Log to wandb
                if cfg.wandb.mode != "disabled":
                    wandb.log({
                        'epoch': epoch,
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy'],
                        'val/mae': val_metrics['mae']
                    })

                # Get metric for checkpointing
                metric_key = cfg.train.checkpoint_metric.replace('val/', '')
                metric_value = val_metrics[metric_key]

                # Save checkpoint
                checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_value=metric_value,
                    is_last=(epoch == cfg.train.max_epochs)
                )

                # Early stopping check
                if is_better_fn(metric_value, best_metric):
                    best_metric = metric_value
                    patience_counter = 0
                    print(f"  New best {cfg.train.early_stopping_metric}: {best_metric:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{cfg.train.early_stopping_patience})")

                # Check early stopping
                if epoch >= cfg.train.min_epochs and patience_counter >= cfg.train.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            # Step scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

    # Test evaluation
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)

    # Load best checkpoint
    best_checkpoint_path = checkpoint_manager.get_best_checkpoint_path()
    if best_checkpoint_path is not None:
        print(f"\nLoading best checkpoint: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Best checkpoint epoch: {checkpoint['epoch']}")
        print(f"  Best checkpoint metric: {checkpoint['metric_value']:.4f}")

    # Run test evaluation
    test_metrics = test_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"\nTest Results:")
    print(f"  Test loss: {test_metrics['loss']:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    # print(f"  Test Precision: {test_metrics['precision']:.4f}")
    # print(f"  Test Recall: {test_metrics['recall']:.4f}")
    # print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")

    # Log test metrics to wandb
    if cfg.wandb.mode != "disabled":
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/accuracy': test_metrics['accuracy'],
            'test/mae': test_metrics['mae']
        })

        # Log best checkpoint as artifact
        if best_checkpoint_path is not None:
            artifact = wandb.Artifact(
                name=f"{cfg.experiment_name}_best_model",
                type="model",
                description=f"Best model checkpoint (epoch {checkpoint['epoch']})"
            )
            artifact.add_file(str(best_checkpoint_path))
            wandb.log_artifact(artifact)

    # Cleanup
    if cfg.wandb.mode != "disabled":
        wandb.finish()

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


# ============================================================================
# Hydra Entry Point
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra entry point.

    Args:
        cfg: Hydra configuration (automatically loaded)
    """
    try:
        train(cfg)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
