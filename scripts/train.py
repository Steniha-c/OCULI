"""
Model Training Script for Diabetic Retinopathy Classification
============================================================

Trains an EfficientNet-B0 backbone for 5-grade diabetic retinopathy classification.
Supports:
- Local Windows development with CPU or NVIDIA GPU
- Google Colab GPU accelerated training
- Automatic mixed precision (AMP)
- Quadratic Weighted Kappa (QWK) and multi-class accuracy tracking
- Modular checkpoint saving (best model & last state)
- Dry-run verification with synthetic fundus generation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

# Allow script execution from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oculi.data.dataset import create_dataloaders
from oculi.models.backbone import build_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("oculi.train")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> float:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Validate the model on the validation split.

    Returns:
        tuple: (average_loss, accuracy, quadratic_weighted_kappa)
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    avg_loss = total_loss / max(1, total_batches)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))

    try:
        qwk = float(cohen_kappa_score(all_targets, all_preds, weights="quadratic"))
    except Exception:
        qwk = 0.0

    return avg_loss, accuracy, qwk


def run_training(args: argparse.Namespace) -> None:
    """Main training orchestration."""
    config = load_config(args.config)
    seed = config["project"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Hyperparameters
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    epochs = args.epochs if args.epochs else train_cfg.get("epochs", 20)
    batch_size = args.batch_size if args.batch_size else train_cfg.get("batch_size", 16)
    lr = args.lr if args.lr else train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    image_size = data_cfg.get("image_size", 224)
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / train_cfg.get("checkpoint_name", "best_model.pth")

    # Datasets
    train_csv = args.train_csv or data_cfg.get("train_csv")
    val_csv = args.val_csv or data_cfg.get("val_csv")
    image_dir = args.image_dir or data_cfg.get("image_dir")

    num_workers = train_cfg.get("num_workers", 0)
    if sys.platform == "win32":
        # Safe default for Windows multiprocessing
        num_workers = min(num_workers, 0)

    is_dry_run = args.dry_run
    dummy_samples = args.synthetic_samples if is_dry_run else 0

    if not is_dry_run and not (train_csv and Path(train_csv).exists()):
        logger.warning(
            f"Training CSV not found at '{train_csv}'. "
            "Switching automatically to dry-run mode with synthetic fundus data. "
            "To train on real clinical data, provide --train-csv and --image-dir."
        )
        is_dry_run = True
        dummy_samples = 20

    if is_dry_run:
        epochs = min(epochs, 2)
        logger.info(f"Executing DRY-RUN verification: {dummy_samples} synthetic samples, {epochs} epochs.")

    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv if not is_dry_run else None,
        val_csv=val_csv if not is_dry_run else None,
        image_dir=image_dir if not is_dry_run else None,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        dummy_samples=dummy_samples,
    )

    logger.info(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")

    # Initialize model
    model = build_model(
        num_classes=model_cfg.get("num_classes", 5),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
        device=device,
    )

    # Optimizer, Loss, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision scaler (if CUDA available)
    scaler = torch.cuda.amp.GradScaler() if (train_cfg.get("mixed_precision", True) and device.type == "cuda") else None

    best_qwk = -1.0
    best_loss = float("inf")

    logger.info("=" * 65)
    logger.info(f"Starting Training: {epochs} Epochs | Batch Size: {batch_size} | LR: {lr}")
    logger.info("=" * 65)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        val_loss, val_acc, val_qwk = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val QWK: {val_qwk:.4f}"
        )

        # Checkpoint if validation metric improves
        if val_qwk > best_qwk or (val_qwk == best_qwk and val_loss < best_loss):
            best_qwk = val_qwk
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_qwk": val_qwk,
                    "config": config,
                },
                best_checkpoint_path,
            )
            logger.info(f"  >>> Saved new best model checkpoint to: {best_checkpoint_path}")

    logger.info("=" * 65)
    logger.info(f"Training run completed! Best Val QWK: {best_qwk:.4f} | Checkpoint: {best_checkpoint_path}")
    logger.info("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Oculi Diabetic Retinopathy Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda')")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to validation CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to retinal images folder")
    parser.add_argument("--dry-run", action="store_true", help="Run short verification with synthetic samples")
    parser.add_argument("--synthetic-samples", type=int, default=16, help="Sample count for dry-run")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
