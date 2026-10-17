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
    - Strict validation of dataset inputs, labels, and file existence.
    - Explicit synthetic mode for software testing and dry runs only.
    - Raises clear errors on missing, unreadable, or corrupt files (no silent fallback).
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
        check_images_exist: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform if transform else get_transforms(image_size, is_train=False)
        self.crop_fov = crop_fov
        self.circular_mask = circular_mask
        self.quality_filter = quality_filter
        self.image_size = image_size
        self.synthetic_samples = synthetic_samples
        self.check_images_exist = check_images_exist

        self.samples: List[Dict[str, Any]] = []

        if synthetic_samples > 0:
            # Generate synthetic sample registry exclusively when explicitly requested
            for i in range(synthetic_samples):
                dummy_grade = i % 5
                self.samples.append({
                    "id": f"synthetic_{i:04d}",
                    "path": None,
                    "label": dummy_grade,
                    "is_synthetic": True,
                })
        else:
            if df_or_csv is None:
                raise ValueError(
                    "Missing dataset input: 'df_or_csv' must be provided when synthetic_samples=0."
                )

            if isinstance(df_or_csv, (str, Path)):
                csv_path = Path(df_or_csv)
                if not csv_path.exists():
                    raise FileNotFoundError(f"Annotation CSV file not found: '{csv_path}'.")
                df = pd.read_csv(csv_path)
            elif isinstance(df_or_csv, pd.DataFrame):
                df = df_or_csv.copy()
            else:
                raise TypeError(
                    f"Unsupported df_or_csv type: {type(df_or_csv)}. Expected Path, str, or DataFrame."
                )

            self._parse_dataframe(df)

    def _parse_dataframe(self, df: pd.DataFrame) -> None:
        """Parse and strictly validate annotations DataFrame."""
        if df.empty:
            raise ValueError("Dataset is empty. The provided CSV / DataFrame contains no rows.")

        id_candidates = ["id_code", "image_id", "image", "filename", "id"]
        id_col = None
        for candidate in id_candidates:
            if candidate in df.columns:
                id_col = candidate
                break

        label_candidates = ["diagnosis", "level", "label", "grade", "target", "dr_grade"]
        label_col = None
        for candidate in label_candidates:
            if candidate in df.columns:
                label_col = candidate
                break

        if id_col is None or label_col is None:
            raise ValueError(
                f"Dataset missing required columns. Could not detect image-ID column "
                f"(supported: {id_candidates}) or label column (supported: {label_candidates}) "
                f"in DataFrame columns: {df.columns.tolist()}."
            )

        # Check for missing/null/blank image IDs
        if df[id_col].isna().any():
            null_id_indices = df[df[id_col].isna()].index.tolist()
            raise ValueError(
                f"Dataset contains null or missing image IDs at row index(es): {null_id_indices[:5]}."
            )

        blank_id_mask = df[id_col].astype(str).str.strip() == ""
        if blank_id_mask.any():
            blank_indices = df[blank_id_mask].index.tolist()
            raise ValueError(
                f"Dataset contains blank/empty image IDs at row index(es): {blank_indices[:5]}."
            )

        # Check for missing/null labels
        if df[label_col].isna().any():
            null_label_indices = df[df[label_col].isna()].index.tolist()
            raise ValueError(
                f"Dataset contains null or missing labels at row index(es): {null_label_indices[:5]}."
            )

        # Validate label values: must be integers in range [0, 4]
        for row_idx, raw_val in df[label_col].items():
            if isinstance(raw_val, (bool, np.bool_)):
                raise ValueError(
                    f"Invalid boolean label '{raw_val}' found at row {row_idx} in column '{label_col}'. "
                    "Labels must be integer DR severity grades in range 0–4."
                )
            try:
                val_float = float(raw_val)
                if not val_float.is_integer():
                    raise ValueError
                int_label = int(val_float)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid non-integer label '{raw_val}' found at row {row_idx} in column '{label_col}'. "
                    "Labels must be integer DR severity grades in range 0–4."
                )

            if int_label < 0 or int_label > 4:
                raise ValueError(
                    f"Label value {int_label} at row {row_idx} in column '{label_col}' is outside the "
                    "supported range 0–4 (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR)."
                )

        # Check image directory if specified
        if self.image_dir is not None:
            if not self.image_dir.exists() or not self.image_dir.is_dir():
                raise FileNotFoundError(
                    f"Image directory not found or is not a directory: '{self.image_dir}'."
                )

        # Verify that all referenced images exist
        missing_images = []
        supported_extensions = ["", ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

        for row_idx, row in df.iterrows():
            img_id = str(row[id_col]).strip()
            int_label = int(float(row[label_col]))

            # Locate file path
            img_path = None
            if self.image_dir:
                for ext in supported_extensions:
                    candidate = self.image_dir / f"{img_id}{ext}"
                    if candidate.exists() and candidate.is_file():
                        img_path = candidate
                        break
            else:
                p = Path(img_id)
                if p.exists() and p.is_file():
                    img_path = p

            if self.check_images_exist and img_path is None:
                missing_images.append(img_id)
            else:
                self.samples.append({
                    "id": img_id,
                    "path": str(img_path) if img_path else None,
                    "label": int_label,
                    "is_synthetic": False,
                })

        if self.check_images_exist and missing_images:
            search_loc = f"directory '{self.image_dir}'" if self.image_dir else "specified paths"
            raise FileNotFoundError(
                f"Referenced image file(s) not found in {search_loc}. "
                f"Missing {len(missing_images)} of {len(df)} images. "
                f"First missing image IDs: {missing_images[:10]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        label = item["label"]
        img_id = item["id"]

        if item.get("is_synthetic"):
            # Synthetic image generated only when explicitly requested (dry-run/testing)
            img_rgb = create_synthetic_fundus_image(
                size=self.image_size, grade=label, seed=idx
            )
        else:
            img_path = item.get("path")
            if img_path is None or not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Image file not found for sample ID '{img_id}' at path: '{img_path}'. "
                    "Synthetic image fallback is disabled in real-data mode."
                )

            bgr = cv2.imread(img_path)
            if bgr is None:
                raise ValueError(
                    f"Failed to read or decode fundus image for sample ID '{img_id}' at path: '{img_path}'. "
                    "The file may be corrupt, empty, or an unsupported image format. "
                    "Synthetic image fallback is disabled in real-data mode."
                )
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
