"""
Regression and Integration Tests for DRPredictor and Preprocessing Alignment
=============================================================================

Validates:
1. Inference preprocessing mathematically and structurally matches training preprocessing:
   - FOV cropping (`detect_and_crop_fov`)
   - Circular aperture masking (`apply_circular_mask`)
   - ImageNet resizing and normalization (mean/std)
2. Empty, dark, or invalid crops are rejected safely with clear ValueError.
3. Grad-CAM uses the exact same preprocessed tensor as inference.
4. End-to-end prediction with saved checkpoint succeeds.
5. Overriding crop_fov and circular_mask options functions as expected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from oculi.data.dataset import create_synthetic_fundus_image, get_transforms
from oculi.models.predictor import DRPredictor
from oculi.preprocessing.quality import apply_circular_mask, detect_and_crop_fov


class TestInferencePreprocessingAlignment:
    """Tests confirming inference preprocessing matches the training pipeline."""

    def test_inference_preprocessing_matches_training_pipeline(self) -> None:
        """
        Verify that DRPredictor.preprocess_image applies FOV cropping, circular masking,
        resizing to 224x224, and ImageNet normalization in the exact same sequence
        as RetinopathyDataset.
        """
        # Generate raw synthetic fundus photograph
        raw_img = create_synthetic_fundus_image(size=320, grade=2, seed=99)

        # 1. Pipeline from training (dataset.py)
        img_cropped, _ = detect_and_crop_fov(raw_img)
        img_masked = apply_circular_mask(img_cropped)
        training_transform = get_transforms(image_size=224, is_train=False)
        expected_tensor = training_transform(img_masked)

        # 2. Pipeline from inference (predictor.py)
        predictor = DRPredictor(checkpoint_path=None, crop_fov=True, circular_mask=True, device="cpu")
        actual_tensor, actual_prep_rgb = predictor.preprocess_image(raw_img)

        # Preprocessed RGB intermediate arrays must match identically
        assert np.array_equal(actual_prep_rgb, img_masked), (
            "Preprocessed RGB image in DRPredictor does not match training preprocessing."
        )

        # Output tensor must have shape (1, 3, 224, 224)
        assert actual_tensor.shape == (1, 3, 224, 224), (
            f"Expected shape (1, 3, 224, 224), got {actual_tensor.shape}"
        )

        # Tensor values must be identical within floating point tolerance
        assert torch.allclose(actual_tensor.squeeze(0), expected_tensor, atol=1e-6), (
            "Normalized tensor in DRPredictor does not match training transform tensor."
        )

    def test_invalid_and_empty_crops_raise_value_error(self) -> None:
        """Verify that empty, too small, or completely dark images are rejected with ValueError."""
        predictor = DRPredictor(checkpoint_path=None, device="cpu")

        # Completely black image (no illuminated retinal tissue > 15)
        black_img = np.zeros((256, 256, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="no illuminated retinal tissue"):
            predictor.preprocess_image(black_img)

        # Empty image array
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty or invalid"):
            predictor.preprocess_image(empty_img)

        # Very small/degenerate array (< 10x10)
        tiny_img = np.zeros((5, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty or invalid"):
            predictor.preprocess_image(tiny_img)

    def test_predict_rejects_empty_image(self) -> None:
        """Verify that DRPredictor.predict safely rejects invalid images rather than producing fake predictions."""
        predictor = DRPredictor(checkpoint_path=None, device="cpu")
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="no illuminated retinal tissue"):
            predictor.predict(black_img)

    def test_end_to_end_prediction_with_checkpoint(self) -> None:
        """Verify full screening inference with saved checkpoint best_model.pth."""
        checkpoint_path = Path("checkpoints/best_model.pth")
        chk = str(checkpoint_path) if checkpoint_path.exists() else None

        predictor = DRPredictor(checkpoint_path=chk, device="cpu")
        img = create_synthetic_fundus_image(size=256, grade=2, seed=42)

        result = predictor.predict(img, generate_cam=True)

        assert 0 <= result.predicted_grade <= 4
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 5
        assert result.preprocessed_image is not None
        assert result.preprocessed_image.shape[2] == 3

        # Grad-CAM heatmap and overlay checks
        assert result.cam_heatmap is not None
        assert result.cam_overlay is not None
        assert result.cam_heatmap.shape == result.preprocessed_image.shape[:2]
        assert result.cam_overlay.shape == result.preprocessed_image.shape
        assert result.metadata["crop_fov"] is True
        assert result.metadata["circular_mask"] is True

    def test_override_preprocessing_flags(self) -> None:
        """Verify that crop_fov and circular_mask can be overridden per prediction call."""
        predictor = DRPredictor(checkpoint_path=None, crop_fov=True, circular_mask=True, device="cpu")
        raw_img = create_synthetic_fundus_image(size=256, grade=0, seed=10)

        # Override to disable crop and mask
        tensor_raw, prep_raw = predictor.preprocess_image(raw_img, crop_fov=False, circular_mask=False)
        assert prep_raw.shape == raw_img.shape
        assert np.array_equal(prep_raw, raw_img)


class TestGradCAMOptimizationAndCalibration:
    """Tests verifying Grad-CAM single forward pass optimization and temperature calibration."""

    def test_grad_cam_single_forward_pass_eliminates_redundancy(self) -> None:
        """
        Verify that DRPredictor.predict() executes the model forward pass exactly ONCE
        even when generate_cam=True, eliminating the redundant second forward pass.
        """
        predictor = DRPredictor(checkpoint_path=None, device="cpu")
        img = create_synthetic_fundus_image(size=256, grade=2, seed=77)

        forward_call_count = 0
        original_forward = predictor.model.forward

        def counting_forward(*args, **kwargs):
            nonlocal forward_call_count
            forward_call_count += 1
            return original_forward(*args, **kwargs)

        predictor.model.forward = counting_forward

        result = predictor.predict(img, generate_cam=True)

        # Must be called exactly once
        assert forward_call_count == 1, (
            f"Expected exactly 1 model forward pass, but recorded {forward_call_count}."
        )
        assert result.cam_heatmap is not None
        assert result.cam_overlay is not None
        assert 0.0 <= result.cam_heatmap.min()
        assert result.cam_heatmap.max() <= 1.0

    def test_grad_cam_generate_heatmap_precomputed_logits_equivalence(self) -> None:
        """
        Verify that GradCAM.generate_heatmap produces identical heatmaps whether
        passed precomputed logits or computing logits internally.
        """
        predictor = DRPredictor(checkpoint_path=None, device="cpu")
        img = create_synthetic_fundus_image(size=256, grade=1, seed=88)
        tensor, _ = predictor.preprocess_image(img)

        # Run with precomputed logits
        from oculi.models.predictor import GradCAM

        cam_pre = GradCAM(predictor.model, predictor.model.target_cam_layer)
        with torch.set_grad_enabled(True):
            logits = predictor.model(tensor)
        heatmap_pre = cam_pre.generate_heatmap(
            input_tensor=tensor, target_class=1, image_size=(224, 224), logits=logits
        )
        cam_pre.close()

        # Run without precomputed logits (standalone evaluation)
        cam_standalone = GradCAM(predictor.model, predictor.model.target_cam_layer)
        heatmap_standalone = cam_standalone.generate_heatmap(
            input_tensor=tensor, target_class=1, image_size=(224, 224), logits=None
        )
        cam_standalone.close()

        # Both heatmaps must be identical
        max_diff = float(np.max(np.abs(heatmap_pre - heatmap_standalone)))
        assert max_diff < 1e-5, f"Heatmaps differed by {max_diff}"

    def test_temperature_scaling_updates_probabilities_without_model_reload(self) -> None:
        """
        Verify that set_temperature() dynamically scales prediction probabilities
        without re-instantiating the PyTorch model or altering raw logits.
        """
        predictor = DRPredictor(checkpoint_path=None, device="cpu")
        img = create_synthetic_fundus_image(size=256, grade=2, seed=42)

        model_id_initial = id(predictor.model)

        # Prediction at default T = 1.0
        res_t1 = predictor.predict(img, generate_cam=False)
        assert abs(sum(res_t1.probabilities.values()) - 1.0) < 1e-4

        # Update to softer distribution T = 2.0 without reloading model
        predictor.set_temperature(2.0)
        assert id(predictor.model) == model_id_initial, "Model instance changed unexpectedly"

        res_t2 = predictor.predict(img, generate_cam=False)
        assert abs(sum(res_t2.probabilities.values()) - 1.0) < 1e-4

        # Raw logits must be identical
        assert np.allclose(res_t1.raw_logits, res_t2.raw_logits, atol=1e-6)

        # With T=2.0, max probability should be softened compared to T=1.0
        assert res_t2.confidence < res_t1.confidence, (
            f"Expected T=2.0 confidence ({res_t2.confidence:.4f}) to be softened "
            f"compared to T=1.0 confidence ({res_t1.confidence:.4f})"
        )

        # Update to sharper distribution T = 0.5
        predictor.set_temperature(0.5)
        res_t05 = predictor.predict(img, generate_cam=False)
        assert res_t05.confidence > res_t1.confidence, (
            f"Expected T=0.5 confidence ({res_t05.confidence:.4f}) to be sharper "
            f"than T=1.0 confidence ({res_t1.confidence:.4f})"
        )

    def test_predict_temperature_parameter_override(self) -> None:
        """Verify that passing temperature directly into predict() overrides the scaler for that call."""
        predictor = DRPredictor(checkpoint_path=None, temperature=1.0, device="cpu")
        img = create_synthetic_fundus_image(size=256, grade=3, seed=12)

        res_default = predictor.predict(img, generate_cam=False)
        res_override = predictor.predict(img, generate_cam=False, temperature=2.0)

        assert res_override.metadata["temperature"] == 2.0
        assert res_override.confidence < res_default.confidence
        # Default scaler remains 1.0
        assert predictor.scaler.temperature == 1.0

    def test_streamlit_get_predictor_caching(self) -> None:
        """Verify that get_predictor() in streamlit_app caches by checkpoint path."""
        from app.streamlit_app import get_predictor

        p1 = get_predictor("checkpoints/best_model.pth")
        p2 = get_predictor("checkpoints/best_model.pth")

        # Must return the exact same cached instance
        assert p1 is p2, "get_predictor failed to return the cached DRPredictor instance"

        # Updating temperature should modify p1 in place without invalidating model
        p1.set_temperature(1.7)
        assert p2.scaler.temperature == 1.7

