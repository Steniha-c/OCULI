"""
Inference Pipeline, Confidence Calibration, Grad-CAM, and Clinical Triage
========================================================================

Orchestrates:
1. Retinal image preprocessing & PyTorch tensor standardization.
2. EfficientNet-B0 inference with temperature calibration.
3. Grad-CAM visual explainability (identifying retinal lesions like hemorrhages,
   exudates, and microaneurysms that drive model attention).
4. Clinical triage recommendation engine based on international DR guidelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from oculi.models.backbone import OculiEfficientNetB0, build_model
from oculi.preprocessing.quality import (
    apply_circular_mask,
    detect_and_crop_fov,
    to_numpy_rgb,
)

logger = logging.getLogger(__name__)

# Standard DR class names according to the International Clinical Diabetic Retinopathy Scale
DR_CLASSES: dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


@dataclass
class TriageDecision:
    """Clinical triage decision generated from DR severity assessment."""

    grade: int
    grade_name: str
    is_referable: bool
    action: str
    urgency_timeframe: str
    priority: str
    clinical_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "grade_name": self.grade_name,
            "is_referable": self.is_referable,
            "action": self.action,
            "urgency_timeframe": self.urgency_timeframe,
            "priority": self.priority,
            "clinical_recommendation": self.clinical_recommendation,
        }


@dataclass
class PredictionResult:
    """Consolidated prediction, calibration, explainability, and triage results."""

    predicted_grade: int
    predicted_class_name: str
    confidence: float
    probabilities: dict[str, float]
    raw_logits: list[float]
    is_referable: bool
    triage: TriageDecision
    cam_heatmap: Optional[np.ndarray] = None
    cam_overlay: Optional[np.ndarray] = None
    preprocessed_image: Optional[np.ndarray] = None
    model_source: str = "ImageNet-Pretrained Prototype (Uncalibrated)"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert prediction results into a JSON-serializable dictionary."""
        return {
            "predicted_grade": self.predicted_grade,
            "predicted_class_name": self.predicted_class_name,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "is_referable": self.is_referable,
            "triage": self.triage.to_dict(),
            "model_source": self.model_source,
            "metadata": self.metadata,
        }


class TemperatureScaler:
    """Post-hoc confidence calibration using temperature scaling."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(1e-4, float(temperature))

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by temperature."""
        return logits / self.temperature

    def compute_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute softmax probabilities with temperature scaling."""
        scaled = self.scale_logits(logits)
        return F.softmax(scaled, dim=-1)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Generates visual explanations highlighting regions that influenced the model's
    classification (e.g. vascular abnormalities, exudates, hemorrhages).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        logits: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate 2D normalized Grad-CAM heatmap for a target class.

        Args:
            input_tensor: Tensor of shape (1, 3, H, W). Required if logits is None.
            target_class: Class index to explain. If None, uses top predicted class.
            image_size: Output spatial resolution (width, height).
            logits: Optional precomputed unnormalized logits from the forward pass.
                When provided, skips the redundant model forward pass.

        Returns:
            np.ndarray: 2D float array in range [0, 1].
        """
        if logits is None:
            if input_tensor is None:
                raise ValueError(
                    "Either 'input_tensor' or precomputed 'logits' must be provided to generate_heatmap."
                )
            self.model.eval()
            self.model.zero_grad()
            logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not triggered
            return np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: (1, C, H', W')
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(pooled_gradients * self.activations, dim=1).squeeze(0)

        # Apply ReLU to retain features that positively correlate with the target class
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to desired image resolution
        heatmap_resized = cv2.resize(cam_np, image_size, interpolation=cv2.INTER_LINEAR)
        return heatmap_resized.astype(np.float32)

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend a Grad-CAM heatmap onto the original RGB fundus image.

        Args:
            image_rgb: uint8 RGB fundus image of shape (H, W, 3).
            heatmap: float array of shape (H, W) in range [0, 1].
            alpha: Transparency weight for heatmap overlay.
            colormap: OpenCV colormap (default JET).

        Returns:
            np.ndarray: Blended uint8 RGB image.
        """
        h, w = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255.0 * heatmap_resized)

        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return blended

    def close(self) -> None:
        """Remove forward and backward hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class TriageEngine:
    """
    Translates DR severity predictions into actionable clinical recommendations
    aligned with the International Council of Ophthalmology (ICO) criteria.
    """

    def __init__(self, referable_threshold: int = 2) -> None:
        self.referable_threshold = referable_threshold

    def evaluate(self, grade: int, confidence: float) -> TriageDecision:
        grade = int(np.clip(grade, 0, 4))
        grade_name = DR_CLASSES.get(grade, f"Grade {grade}")
        is_referable = grade >= self.referable_threshold

        triage_policies: dict[int, dict[str, str]] = {
            0: {
                "action": "Routine Annual Rescreening",
                "timeframe": "12 Months",
                "priority": "Routine",
                "clinical_recommendation": (
                    "No diabetic retinopathy signs detected. Patient should continue "
                    "routine annual fundus photography and adhere to primary diabetes management."
                ),
            },
            1: {
                "action": "Clinical Surveillance / Monitoring",
                "timeframe": "6 - 12 Months",
                "priority": "Low",
                "clinical_recommendation": (
                    "Mild non-proliferative changes (microaneurysms only). Re-screen in 6-12 months. "
                    "Review systemic blood pressure, HbA1c, and lipid targets with primary care physician."
                ),
            },
            2: {
                "action": "Ophthalmic Referral (Routine)",
                "timeframe": "2 - 3 Months",
                "priority": "Moderate (Referable DR)",
                "clinical_recommendation": (
                    "Moderate non-proliferative retinopathy (more than microaneurysms, blot hemorrhages). "
                    "Refer to comprehensive ophthalmology for dilated stereoscopic biomicroscopy and OCT assessment."
                ),
            },
            3: {
                "action": "Ophthalmic Referral (High Priority)",
                "timeframe": "2 - 4 Weeks",
                "priority": "High (Referable DR)",
                "clinical_recommendation": (
                    "Severe non-proliferative retinopathy (4-2-1 rule criteria). Significant risk of "
                    "rapid progression to proliferative disease. Expedited specialist assessment required."
                ),
            },
            4: {
                "action": "Urgent Ophthalmic Referral",
                "timeframe": "Within 1 Week",
                "priority": "Urgent (Sight-Threatening DR)",
                "clinical_recommendation": (
                    "Proliferative diabetic retinopathy (neovascularization or vitreous/preretinal hemorrhage). "
                    "Urgent panretinal photocoagulation (PRP) or anti-VEGF therapy evaluation required."
                ),
            },
        }

        policy = triage_policies[grade]
        return TriageDecision(
            grade=grade,
            grade_name=grade_name,
            is_referable=is_referable,
            action=policy["action"],
            urgency_timeframe=policy["timeframe"],
            priority=policy["priority"],
            clinical_recommendation=policy["clinical_recommendation"],
        )


class DRPredictor:
    """
    End-to-end predictor for Diabetic Retinopathy screening prototype.

    Handles:
    - Image loading and normalization
    - Model forward inference
    - Post-hoc temperature calibration
    - Grad-CAM saliency extraction
    - Clinical referral triage
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        image_size: int = 224,
        temperature: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        crop_fov: bool = True,
        circular_mask: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.crop_fov = crop_fov
        self.circular_mask = circular_mask
        self.scaler = TemperatureScaler(temperature=temperature)
        self.triage_engine = TriageEngine(referable_threshold=2)

        # Track checkpoint status
        self.checkpoint_path = checkpoint_path
        self.is_custom_checkpoint = checkpoint_path is not None and Path(checkpoint_path).exists()

        if self.is_custom_checkpoint:
            self.model = build_model(
                num_classes=5,
                pretrained=False,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            self.model_source = f"Custom Checkpoint ({Path(self.checkpoint_path).name})"
            logger.info(f"Loaded trained checkpoint: {self.checkpoint_path}")
        else:
            # Prototype initialization using ImageNet pretrained weights
            self.model = build_model(
                num_classes=5,
                pretrained=True,
                device=self.device,
            )
            self.model_source = "ImageNet-Pretrained Prototype (Uncalibrated Research Baseline)"
            logger.info("Initialized uncalibrated prototype model with ImageNet features.")

        self.model.eval()

        # ImageNet normalization pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def set_temperature(self, temperature: float) -> None:
        """
        Update confidence calibration temperature without re-instantiating the model or reloading weights.

        Args:
            temperature: Divisor for logits (>0). T > 1 softens probabilities; T < 1 sharpens them.
        """
        self.scaler = TemperatureScaler(temperature=temperature)

    def preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        crop_fov: Optional[bool] = None,
        circular_mask: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess an input fundus image using the exact pipeline from training:
        1. Standardize input to uint8 RGB NumPy array.
        2. Detect and crop the circular retinal Field-of-View (FOV).
        3. Apply circular aperture mask to zero out corner artifacts.
        4. Validate that the resulting crop is non-empty and contains valid retinal tissue.
        5. Apply standard ImageNet resizing and normalization to PyTorch tensor.

        Args:
            image_input: Retinal image (path, PIL Image, or RGB NumPy array).
            crop_fov: Optional override for FOV cropping (defaults to self.crop_fov).
            circular_mask: Optional override for circular aperture masking (defaults to self.circular_mask).

        Returns:
            Tuple[torch.Tensor, np.ndarray]:
                - Normalized tensor of shape (1, 3, H, W) on the target device.
                - Intermediate preprocessed RGB NumPy array (cropped and masked).
        """
        apply_crop = self.crop_fov if crop_fov is None else crop_fov
        apply_mask = self.circular_mask if circular_mask is None else circular_mask

        image_rgb = to_numpy_rgb(image_input)
        if image_rgb.size == 0 or image_rgb.shape[0] < 10 or image_rgb.shape[1] < 10:
            raise ValueError(
                f"Input fundus image is empty or invalid: shape {image_rgb.shape}."
            )

        # 1. Retinal Field-of-View (FOV) Cropping
        if apply_crop:
            cropped_rgb, bbox = detect_and_crop_fov(image_rgb)
            if (
                cropped_rgb.size == 0
                or cropped_rgb.shape[0] < 10
                or cropped_rgb.shape[1] < 10
            ):
                raise ValueError(
                    f"Invalid or empty crop detected: retinal FOV segmentation resulted in invalid dimensions "
                    f"{cropped_rgb.shape[:2]}."
                )

            # Quality check: ensure image contains illuminated retinal tissue
            gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            if not np.any(gray > 15):
                raise ValueError(
                    "Invalid retinal Field-of-View: image contains no illuminated retinal tissue "
                    "(entirely dark or empty image)."
                )

            image_rgb = cropped_rgb

        # 2. Circular Aperture Masking
        if apply_mask:
            image_rgb = apply_circular_mask(image_rgb)

        # 3. Standard PyTorch ImageNet Transformation
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        return tensor, image_rgb

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        generate_cam: bool = True,
        target_class: Optional[int] = None,
        crop_fov: Optional[bool] = None,
        circular_mask: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> PredictionResult:
        """
        Execute full screening pipeline on an input fundus image.

        Args:
            image_input: Retinal image (path, PIL Image, or RGB NumPy array).
            generate_cam: If True, computes Grad-CAM explainability heatmap.
            target_class: Specific class to explain with Grad-CAM (None = top class).
            crop_fov: Optional override for FOV cropping.
            circular_mask: Optional override for circular aperture masking.
            temperature: Optional override for confidence calibration temperature.

        Returns:
            PredictionResult: Rich inference outcome with triage and CAM overlay.
        """
        raw_rgb = to_numpy_rgb(image_input)
        h_orig, w_orig = raw_rgb.shape[:2]

        # Apply preprocessing matching training (FOV crop + circular mask)
        tensor, preprocessed_rgb = self.preprocess_image(
            image_input=raw_rgb,
            crop_fov=crop_fov,
            circular_mask=circular_mask,
        )
        h_prep, w_prep = preprocessed_rgb.shape[:2]

        # Grad-CAM setup if requested (register hook before forward pass)
        grad_cam: Optional[GradCAM] = None
        if generate_cam:
            grad_cam = GradCAM(self.model, self.model.target_cam_layer)

        # Single model inference pass (with gradients enabled if Grad-CAM requested)
        with torch.set_grad_enabled(generate_cam):
            logits = self.model(tensor)

        # Calculate calibrated probabilities using active scaler
        effective_scaler = (
            self.scaler if temperature is None else TemperatureScaler(temperature=temperature)
        )
        with torch.no_grad():
            probs_tensor = effective_scaler.compute_probabilities(logits).squeeze(0)
            probs = probs_tensor.cpu().numpy()

        pred_grade = int(np.argmax(probs))
        confidence = float(probs[pred_grade])

        prob_dict: dict[str, float] = {
            DR_CLASSES[i]: float(probs[i]) for i in range(len(DR_CLASSES))
        }

        # Generate CAM using precomputed logits (eliminating redundant forward pass)
        cam_heatmap: Optional[np.ndarray] = None
        cam_overlay: Optional[np.ndarray] = None

        if generate_cam and grad_cam is not None:
            cam_class = pred_grade if target_class is None else target_class
            cam_heatmap = grad_cam.generate_heatmap(
                input_tensor=tensor,
                target_class=cam_class,
                image_size=(w_prep, h_prep),
                logits=logits,
            )
            cam_overlay = grad_cam.overlay_heatmap(preprocessed_rgb, cam_heatmap)
            grad_cam.close()

        # Clinical Triage Recommendation
        triage = self.triage_engine.evaluate(grade=pred_grade, confidence=confidence)

        effective_crop = self.crop_fov if crop_fov is None else crop_fov
        effective_mask = self.circular_mask if circular_mask is None else circular_mask

        return PredictionResult(
            predicted_grade=pred_grade,
            predicted_class_name=DR_CLASSES[pred_grade],
            confidence=confidence,
            probabilities=prob_dict,
            raw_logits=logits.detach().squeeze(0).cpu().numpy().tolist(),
            is_referable=triage.is_referable,
            triage=triage,
            cam_heatmap=cam_heatmap,
            cam_overlay=cam_overlay,
            preprocessed_image=preprocessed_rgb,
            model_source=self.model_source,
            metadata={
                "input_resolution": (w_orig, h_orig),
                "preprocessed_resolution": (w_prep, h_prep),
                "model_input_size": (self.image_size, self.image_size),
                "crop_fov": effective_crop,
                "circular_mask": effective_mask,
                "temperature": effective_scaler.temperature,
                "device": str(self.device),
            },
        )
