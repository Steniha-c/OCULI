"""
Unit Tests for Fundus Preprocessing, Quality Verification, and Triage
=====================================================================

Validates:
1. Sharpness / Blur detection (Laplacian variance on green channel).
2. Illumination and exposure checks (adequate, underexposed, overexposed).
3. Field-of-View (FOV) segmentation and border trimming.
4. Circular aperture masking.
5. Clinical Triage decision logic (Grades 0 to 4).
6. Backbone model tensor forward pass shape.
"""

import cv2
import numpy as np
import pytest
import torch

from oculi.data.dataset import create_synthetic_fundus_image
from oculi.models.backbone import OculiEfficientNetB0
from oculi.models.predictor import TriageEngine
from oculi.preprocessing.quality import (
    apply_circular_mask,
    assess_fundus_quality,
    compute_blur_score,
    compute_illumination_metrics,
    detect_and_crop_fov,
)


class TestFundusQualityVerification:
    """Test suite verifying retinal fundus image quality verification algorithms."""

    def test_sharp_vs_blurred_detection(self) -> None:
        """Verify that blurred images trigger blur warnings and low scores."""
        # Generate sharp synthetic fundus image
        sharp_img = create_synthetic_fundus_image(size=256, grade=2, seed=42)
        sharp_score = compute_blur_score(sharp_img)

        # Create heavily blurred counterpart
        blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), sigmaX=8.0)
        blurred_score = compute_blur_score(blurred_img)

        # Sharp image must have significantly higher Laplacian variance than blurred
        assert sharp_score > blurred_score * 3.0, (
            f"Expected sharp score ({sharp_score:.2f}) to be significantly greater "
            f"than blurred score ({blurred_score:.2f})"
        )

        # Assess quality using threshold
        report_blurred = assess_fundus_quality(blurred_img, blur_threshold=sharp_score * 0.5)
        assert report_blurred.is_sharp is False
        assert report_blurred.is_acceptable is False
        assert any("blurred" in w.lower() for w in report_blurred.warnings)

    def test_underexposed_image_detection(self) -> None:
        """Verify that very dark/underexposed images are flagged."""
        dark_img = np.zeros((200, 200, 3), dtype=np.uint8)
        dark_img[50:150, 50:150] = 12  # Below min intensity threshold 30.0

        report = assess_fundus_quality(dark_img, min_mean_intensity=30.0)
        assert report.illumination_status == "underexposed"
        assert report.is_acceptable is False
        assert any("underexposed" in w.lower() for w in report.warnings)

    def test_overexposed_image_detection(self) -> None:
        """Verify that saturated/glare images are flagged."""
        bright_img = np.full((200, 200, 3), 245, dtype=np.uint8)

        report = assess_fundus_quality(bright_img, max_mean_intensity=220.0)
        assert report.illumination_status == "overexposed"
        assert report.is_acceptable is False
        assert any("overexposed" in w.lower() for w in report.warnings)

    def test_fov_crop_removes_black_margins(self) -> None:
        """Verify that detect_and_crop_fov trims black borders while keeping retina."""
        padded_img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Put a 200x200 fundus circle in the center
        center = (200, 200)
        cv2.circle(padded_img, center, 80, (180, 80, 30), thickness=-1)

        cropped, bbox = detect_and_crop_fov(padded_img, threshold=15)

        h_orig, w_orig = padded_img.shape[:2]
        h_crop, w_crop = cropped.shape[:2]

        assert h_crop < h_orig
        assert w_crop < w_orig
        assert bbox["ymin"] > 0
        assert bbox["xmin"] > 0

    def test_circular_mask_zeros_corners(self) -> None:
        """Verify that circular mask clears four square corners."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        masked = apply_circular_mask(img)

        # Four outer corners must be set to 0
        assert np.all(masked[0, 0] == 0)
        assert np.all(masked[0, 99] == 0)
        assert np.all(masked[99, 0] == 0)
        assert np.all(masked[99, 99] == 0)
        # Center should remain non-zero
        assert np.all(masked[50, 50] == 200)


class TestClinicalTriageEngine:
    """Test suite verifying referral logic across the 5 DR severity grades."""

    def test_non_referable_grades(self) -> None:
        engine = TriageEngine(referable_threshold=2)

        # Grade 0: Normal
        dec_0 = engine.evaluate(grade=0, confidence=0.95)
        assert dec_0.is_referable is False
        assert dec_0.priority == "Routine"
        assert "12 Months" in dec_0.urgency_timeframe

        # Grade 1: Mild NPDR
        dec_1 = engine.evaluate(grade=1, confidence=0.88)
        assert dec_1.is_referable is False
        assert dec_1.priority == "Low"
        assert "6 - 12 Months" in dec_1.urgency_timeframe

    def test_referable_grades(self) -> None:
        engine = TriageEngine(referable_threshold=2)

        # Grade 2: Moderate NPDR
        dec_2 = engine.evaluate(grade=2, confidence=0.85)
        assert dec_2.is_referable is True
        assert "Moderate" in dec_2.priority
        assert "2 - 3 Months" in dec_2.urgency_timeframe

        # Grade 3: Severe NPDR
        dec_3 = engine.evaluate(grade=3, confidence=0.91)
        assert dec_3.is_referable is True
        assert "High" in dec_3.priority
        assert "2 - 4 Weeks" in dec_3.urgency_timeframe

        # Grade 4: Proliferative DR
        dec_4 = engine.evaluate(grade=4, confidence=0.96)
        assert dec_4.is_referable is True
        assert "Urgent" in dec_4.priority
        assert "1 Week" in dec_4.urgency_timeframe


class TestModelArchitecture:
    """Test suite verifying PyTorch model shapes and forward pass."""

    def test_efficientnet_forward_pass_shape(self) -> None:
        """Verify EfficientNet-B0 backbone outputs (batch_size, 5) logits."""
        model = OculiEfficientNetB0(num_classes=5, pretrained=False)
        model.eval()

        dummy_batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_batch)

        assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
        assert torch.isfinite(output).all(), "Logits must be finite numbers"
