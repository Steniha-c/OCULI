"""
Regression and Input Validation Tests for Oculi Dataset Pipeline
================================================================

Validates:
1. Missing CSV file detection.
2. Missing or incorrect column header detection.
3. Empty dataset rejection.
4. Null/blank image ID rejection.
5. Missing/null label rejection.
6. Non-integer and boolean label rejection.
7. Out-of-bounds label rejection (outside 0-4).
8. Missing image file detection before training.
9. Corrupted/unreadable image rejection in real-data mode (no synthetic fallback).
10. Explicit synthetic dry-run verification mode.
11. Accidental dry-run prevention in scripts/train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from oculi.data.dataset import RetinopathyDataset, create_dataloaders
from scripts.train import run_training


class TestDatasetValidation:
    """Tests covering dataset input validation and error reporting."""

    def test_missing_csv_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify that a non-existent CSV file raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError, match="Annotation CSV file not found"):
            RetinopathyDataset(df_or_csv=non_existent)

    def test_empty_dataset_raises_value_error(self) -> None:
        """Verify that an empty DataFrame or CSV raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Dataset is empty"):
            RetinopathyDataset(df_or_csv=empty_df)

    def test_missing_id_column_raises_value_error(self) -> None:
        """Verify that missing supported image-ID columns raises ValueError."""
        df = pd.DataFrame({"random_col": [1, 2], "diagnosis": [0, 1]})
        with pytest.raises(ValueError, match="Could not detect image-ID column"):
            RetinopathyDataset(df_or_csv=df)

    def test_missing_label_column_raises_value_error(self) -> None:
        """Verify that missing supported label columns raises ValueError."""
        df = pd.DataFrame({"id_code": ["img_01", "img_02"], "notes": ["a", "b"]})
        with pytest.raises(ValueError, match="Could not detect.*label column"):
            RetinopathyDataset(df_or_csv=df)

    def test_null_or_blank_image_id_raises_value_error(self) -> None:
        """Verify that null or whitespace-only image IDs are rejected."""
        # Null ID
        df_null = pd.DataFrame({"id_code": [None, "img_02"], "diagnosis": [0, 1]})
        with pytest.raises(ValueError, match="contains null or missing image IDs"):
            RetinopathyDataset(df_or_csv=df_null)

        # Blank ID
        df_blank = pd.DataFrame({"id_code": ["   ", "img_02"], "diagnosis": [0, 1]})
        with pytest.raises(ValueError, match="contains blank/empty image IDs"):
            RetinopathyDataset(df_or_csv=df_blank)

    def test_null_label_raises_value_error(self) -> None:
        """Verify that missing/null labels are rejected."""
        df_null_label = pd.DataFrame({"id_code": ["img_01", "img_02"], "diagnosis": [0, None]})
        with pytest.raises(ValueError, match="contains null or missing labels"):
            RetinopathyDataset(df_or_csv=df_null_label)

    def test_non_integer_label_raises_value_error(self) -> None:
        """Verify that non-integer labels (floats with decimals, strings, booleans) are rejected."""
        # Float with decimal
        df_float = pd.DataFrame({"id_code": ["img_01"], "diagnosis": [1.5]})
        with pytest.raises(ValueError, match="Invalid non-integer label"):
            RetinopathyDataset(df_or_csv=df_float)

        # String text
        df_str = pd.DataFrame({"id_code": ["img_01"], "diagnosis": ["moderate"]})
        with pytest.raises(ValueError, match="Invalid non-integer label"):
            RetinopathyDataset(df_or_csv=df_str)

        # Boolean
        df_bool = pd.DataFrame({"id_code": ["img_01"], "diagnosis": [True]})
        with pytest.raises(ValueError, match="Invalid boolean label"):
            RetinopathyDataset(df_or_csv=df_bool)

    def test_out_of_range_label_raises_value_error(self) -> None:
        """Verify that labels outside range 0–4 are rejected."""
        df_high = pd.DataFrame({"id_code": ["img_01"], "diagnosis": [5]})
        with pytest.raises(ValueError, match="outside the supported range 0–4"):
            RetinopathyDataset(df_or_csv=df_high)

        df_neg = pd.DataFrame({"id_code": ["img_01"], "diagnosis": [-1]})
        with pytest.raises(ValueError, match="outside the supported range 0–4"):
            RetinopathyDataset(df_or_csv=df_neg)


class TestImageHandlingAndFallbacks:
    """Tests covering physical image presence and strict fallback prevention."""

    def test_missing_referenced_images_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify that missing image files on disk raise FileNotFoundError before training."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # CSV references images that do not exist in img_dir
        df = pd.DataFrame({"id_code": ["retina_001", "retina_002"], "diagnosis": [0, 2]})

        with pytest.raises(FileNotFoundError, match="Referenced image file\\(s\\) not found"):
            RetinopathyDataset(df_or_csv=df, image_dir=img_dir, check_images_exist=True)

    def test_corrupted_image_raises_value_error_without_synthetic_fallback(
        self, tmp_path: Path
    ) -> None:
        """
        Verify that an unreadable or corrupt image file raises ValueError upon loading,
        and is NEVER silently substituted with a synthetic image.
        """
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        corrupt_file = img_dir / "corrupt_fundus.jpg"
        # Write 0 bytes or garbage data that cv2.imread cannot decode
        corrupt_file.write_bytes(b"GARBAGE_NOT_A_VALID_IMAGE_HEADER")

        df = pd.DataFrame({"id_code": ["corrupt_fundus"], "diagnosis": [3]})

        # Initialization passes since file exists on disk
        dataset = RetinopathyDataset(df_or_csv=df, image_dir=img_dir, check_images_exist=True)
        assert len(dataset) == 1

        # Accessing the corrupted item must raise ValueError, NOT silently return a synthetic image
        with pytest.raises(ValueError, match="Failed to read or decode fundus image"):
            _ = dataset[0]

    def test_valid_real_images_load_correctly(self, tmp_path: Path) -> None:
        """Verify that genuine fundus image files load correctly."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create valid mock test images
        img_data = np.zeros((100, 100, 3), dtype=np.uint8)
        img_data[:, :, 0] = 180
        cv2.imwrite(str(img_dir / "img1.png"), img_data)

        df = pd.DataFrame({"id_code": ["img1"], "diagnosis": [1]})
        dataset = RetinopathyDataset(df_or_csv=df, image_dir=img_dir, check_images_exist=True)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["label"].item() == 1
        assert sample["id"] == "img1"
        assert sample["image"].shape == (3, 224, 224)

    def test_explicit_synthetic_mode_succeeds(self) -> None:
        """Verify that explicit synthetic mode works when synthetic_samples > 0."""
        dataset = RetinopathyDataset(synthetic_samples=5)
        assert len(dataset) == 5
        sample = dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["label"].item() in range(5)
        assert "synthetic" in sample["id"]


class TestAccidentalDryRunPrevention:
    """Tests confirming that train.py rejects missing paths and does not switch to dry-run."""

    def test_train_rejects_missing_real_paths(self, tmp_path: Path) -> None:
        """
        Verify that scripts/train.py raises FileNotFoundError when real data paths do not exist,
        rather than silently executing a synthetic dry-run.
        """
        non_existent_csv = str(tmp_path / "non_existent_train.csv")
        non_existent_val = str(tmp_path / "non_existent_val.csv")
        non_existent_dir = str(tmp_path / "non_existent_dir")

        args = argparse.Namespace(
            config="configs/default.yaml",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            device="cpu",
            train_csv=non_existent_csv,
            val_csv=non_existent_val,
            image_dir=non_existent_dir,
            dry_run=False,  # Real data mode requested
            synthetic_samples=4,
        )

        with pytest.raises(FileNotFoundError, match="Training CSV not found"):
            run_training(args)

    def test_train_explicit_dry_run_executes(self) -> None:
        """Verify that explicit --dry-run still executes as intended for software verification."""
        args = argparse.Namespace(
            config="configs/default.yaml",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            device="cpu",
            train_csv=None,
            val_csv=None,
            image_dir=None,
            dry_run=True,  # Explicit dry-run
            synthetic_samples=4,
        )

        # Should complete without error
        run_training(args)
