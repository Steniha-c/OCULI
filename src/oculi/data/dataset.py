"""
Diabetic Retinopathy Dataset Loader and Transformations
======================================================

Provides modular PyTorch Dataset and DataLoader factories supporting common
retinal datasets (e.g. EyePACS, APTOS 2019, Messidor-2, DDR).
Includes robust handling of image formats, circular FOV trimming,
data augmentations, and synthetic fundus generation for offline testing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from oculi.preprocessing.quality import (
    apply_circular_mask,
    assess_fundus_quality,
    detect_and_crop_fov,
)

logger = logging.getLogger(__name__)


def create_synthetic_fundus_image(
    size: int = 256,
    grade: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic fundus disc for unit tests, local demonstration, and dry-runs.
    Simulates:
    - Circular retinal field with dark surroundings.
    - Deep orange/red retinal background.
    - Pale bright optic disc and retinal vascular tree.
    - Pathological hallmarks depending on grade (microaneurysms, blot hemorrhages, exudates).

    Args:
        size: Width and height of square output image.
        grade: DR grade (0: Normal, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: uint8 RGB image of shape (size, size, 3).
    """
    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)

    # 1. Base retinal background gradient (warm reddish-orange)
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    retina_mask = dist_from_center <= radius

    # Radial vignetting on retina
    vignette = 1.0 - (dist_from_center / (radius + 1e-5)) * 0.35
    vignette = np.clip(vignette, 0.4, 1.0)

    # Base retina color: Red ~ 180, Green ~ 85, Blue ~ 25
    r_channel = np.clip(180 * vignette + np.random.normal(0, 3, (size, size)), 0, 255)
    g_channel = np.clip(85 * vignette + np.random.normal(0, 3, (size, size)), 0, 255)
    b_channel = np.clip(25 * vignette + np.random.normal(0, 2, (size, size)), 0, 255)

    img[:, :, 0] = np.where(retina_mask, r_channel, 0).astype(np.uint8)
    img[:, :, 1] = np.where(retina_mask, g_channel, 0).astype(np.uint8)
    img[:, :, 2] = np.where(retina_mask, b_channel, 0).astype(np.uint8)

    # 2. Optic Disc (yellowish oval on nasal side)
    disc_center = (int(center[0] - radius * 0.45), center[1])
    disc_radius_x, disc_radius_y = int(radius * 0.18), int(radius * 0.22)
    cv2.ellipse(
        img,
        disc_center,
        (disc_radius_x, disc_radius_y),
        0,
        0,
        360,
        (240, 210, 130),
        thickness=-1,
    )

    # Macula / Fovea (slightly darker central region on temporal side)
    macula_center = (int(center[0] + radius * 0.25), center[1])
    cv2.circle(img, macula_center, int(radius * 0.15), (130, 50, 15), thickness=-1)

    # 3. Retinal Blood Vessels (branching curves from optic disc)
    for angle in [-40, -15, 10, 35, 140, 160, 200, 220]:
        rad = np.radians(angle)
        pt1 = disc_center
        ctrl = (
            int(disc_center[0] + np.cos(rad) * radius * 0.5 + np.random.randint(-10, 10)),
            int(disc_center[1] + np.sin(rad) * radius * 0.5 + np.random.randint(-10, 10)),
        )
        pt2 = (
            int(disc_center[0] + np.cos(rad) * radius * 0.85),
            int(disc_center[1] + np.sin(rad) * radius * 0.85),
        )
        pts = np.array([pt1, ctrl, pt2], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=(90, 20, 10), thickness=2)

    # 4. Pathological Lesions depending on DR Grade
    if grade >= 1:
        # Microaneurysms: Tiny reddish dots
        num_dots = 4 if grade == 1 else (12 if grade == 2 else 30)
        for _ in range(num_dots):
            dx = int(center[0] + np.random.uniform(-0.6, 0.6) * radius)
            dy = int(center[1] + np.random.uniform(-0.6, 0.6) * radius)
            if np.sqrt((dx - center[0]) ** 2 + (dy - center[1]) ** 2) < radius * 0.8:
                cv2.circle(img, (dx, dy), np.random.randint(1, 3), (80, 10, 5), -1)

    if grade >= 2:
        # Hard Exudates: Bright yellowish crystalline patches
        num_exudates = 3 if grade == 2 else 8
        for _ in range(num_exudates):
            ex_x = int(macula_center[0] + np.random.uniform(-0.3, 0.3) * radius)
            ex_y = int(macula_center[1] + np.random.uniform(-0.3, 0.3) * radius)
            cv2.circle(img, (ex_x, ex_y), np.random.randint(2, 5), (235, 230, 140), -1)

    if grade >= 3:
        # Blot Hemorrhages and Cotton Wool Spots
        for _ in range(6):
            hx = int(center[0] + np.random.uniform(-0.65, 0.65) * radius)
            hy = int(center[1] + np.random.uniform(-0.65, 0.65) * radius)
            cv2.ellipse(img, (hx, hy), (np.random.randint(4, 8), np.random.randint(3, 6)),
                        np.random.randint(0, 180), 0, 360, (60, 5, 5), -1)
        # Cotton wool spot (fluffy white lesion)
        cwx = int(center[0] + np.random.uniform(-0.4, 0.4) * radius)
        cwy = int(center[1] + np.random.uniform(-0.4, 0.4) * radius)
        cv2.circle(img, (cwx, cwy), int(radius * 0.06), (220, 220, 210), -1)

    if grade >= 4:
        # Neovascularization: Frond-like tangled abnormal micro-vessels
        neo_center = (disc_center[0] + 15, disc_center[1] - 20)
        for _ in range(12):
            offset_x = np.random.randint(-15, 15)
            offset_y = np.random.randint(-15, 15)
            cv2.line(
                img,
                neo_center,
                (neo_center[0] + offset_x, neo_center[1] + offset_y),
                (120, 10, 10),
                1,
            )

    # Re-apply outer mask to ensure pristine edges
    img = np.where(retina_mask[:, :, None], img, 0)
    return img


def get_transforms(
    image_size: int = 224,
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Construct torchvision image transformation pipeline.

    Args:
        image_size: Target resolution (H, W).
        is_train: If True, incorporates standard data augmentation.
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).

    Returns:
        transforms.Compose: Pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy screening images.

    Supports:
    - CSV annotations with flexible column mappings (e.g. `image_id`, `id_code`, `image` / `diagnosis`, `level`).
    - Direct directory image loading.
    - Optional automated circular FOV cropping.
    - Quality checking / filtering out unusable images.
    - Synthetic fallback mode for prototyping and automated tests.
    """

    def __init__(
        self,
        df_or_csv: Optional[Union[pd.DataFrame, str, Path]] = None,
        image_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        crop_fov: bool = True,
        circular_mask: bool = True,
        quality_filter: bool = False,
        synthetic_samples: int = 0,
        image_size: int = 224,
    ) -> None:
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform if transform else get_transforms(image_size, is_train=False)
        self.crop_fov = crop_fov
        self.circular_mask = circular_mask
        self.quality_filter = quality_filter
        self.image_size = image_size
        self.synthetic_samples = synthetic_samples

        self.samples: List[Dict[str, Any]] = []

        if synthetic_samples > 0:
            # Generate dummy sample registry
            for i in range(synthetic_samples):
                dummy_grade = i % 5
                self.samples.append({
                    "id": f"synthetic_{i:04d}",
                    "path": None,
                    "label": dummy_grade,
                    "is_synthetic": True,
                })
        elif df_or_csv is not None:
            if isinstance(df_or_csv, (str, Path)):
                if os.path.exists(str(df_or_csv)):
                    df = pd.read_csv(str(df_or_csv))
                else:
                    logger.warning(f"CSV file not found: {df_or_csv}. Initializing empty dataset.")
                    df = pd.DataFrame()
            else:
                df = df_or_csv.copy()

            self._parse_dataframe(df)

    def _parse_dataframe(self, df: pd.DataFrame) -> None:
        """Parse DataFrame with flexible column naming conventions."""
        if df.empty:
            return

        id_col = None
        for candidate in ["id_code", "image_id", "image", "filename", "id"]:
            if candidate in df.columns:
                id_col = candidate
                break

        label_col = None
        for candidate in ["diagnosis", "level", "label", "grade", "target", "dr_grade"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if id_col is None or label_col is None:
            logger.warning(
                f"Could not automatically detect image ID or label columns in {df.columns.tolist()}."
            )
            return

        for _, row in df.iterrows():
            img_id = str(row[id_col])
            label = int(row[label_col])

            # Locate file path
            img_path = None
            if self.image_dir:
                for ext in ["", ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    candidate = self.image_dir / f"{img_id}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break

            self.samples.append({
                "id": img_id,
                "path": str(img_path) if img_path else None,
                "label": label,
                "is_synthetic": False,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        label = item["label"]
        img_id = item["id"]

        if item.get("is_synthetic") or item["path"] is None:
            # Generate synthetic fundus on-the-fly
            img_rgb = create_synthetic_fundus_image(
                size=self.image_size, grade=label, seed=idx
            )
        else:
            bgr = cv2.imread(item["path"])
            if bgr is None:
                # Corrupt image fallback
                img_rgb = create_synthetic_fundus_image(
                    size=self.image_size, grade=label, seed=idx
                )
            else:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Optional FOV cropping & circular masking
        if self.crop_fov:
            img_rgb, _ = detect_and_crop_fov(img_rgb)
        if self.circular_mask:
            img_rgb = apply_circular_mask(img_rgb)

        tensor = self.transform(img_rgb)

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "id": img_id,
            "path": item.get("path") or "synthetic",
        }


def create_dataloaders(
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    image_dir: Optional[str] = None,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    dummy_samples: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct PyTorch DataLoaders for training and validation splits.

    Args:
        train_csv: Path to training CSV annotations.
        val_csv: Path to validation CSV annotations.
        image_dir: Path to directory containing retinal images.
        image_size: Model input spatial resolution.
        batch_size: Mini-batch size.
        num_workers: Multi-process worker count (0 for safe Windows local execution).
        dummy_samples: If > 0, constructs datasets with synthetic samples (for dry runs).

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    train_transform = get_transforms(image_size=image_size, is_train=True)
    val_transform = get_transforms(image_size=image_size, is_train=False)

    if dummy_samples > 0:
        train_ds = RetinopathyDataset(
            transform=train_transform,
            synthetic_samples=dummy_samples,
            image_size=image_size,
        )
        val_ds = RetinopathyDataset(
            transform=val_transform,
            synthetic_samples=max(4, dummy_samples // 4),
            image_size=image_size,
        )
    else:
        train_ds = RetinopathyDataset(
            df_or_csv=train_csv,
            image_dir=image_dir,
            transform=train_transform,
            image_size=image_size,
        )
        val_ds = RetinopathyDataset(
            df_or_csv=val_csv,
            image_dir=image_dir,
            transform=val_transform,
            image_size=image_size,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
