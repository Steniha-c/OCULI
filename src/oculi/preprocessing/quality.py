"""
Fundus Image Quality Assessment and Preprocessing
=================================================

Provides modular checks for retinal photography:
1. Blur / Sharpness estimation via Laplacian variance.
2. Illumination and contrast verification (underexposure / overexposure / glare).
3. Field-of-View (FOV) circle segmentation and black border cropping.
4. Circular aperture masking and Ben Graham's local contrast enhancement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityReport:
    """Diagnostic report describing image quality checks for retinal fundus photography."""

    is_acceptable: bool
    blur_score: float
    is_sharp: bool
    mean_intensity: float
    illumination_status: str  # "adequate", "underexposed", "overexposed"
    fov_ratio: float
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format for logging or JSON serialization."""
        return {
            "is_acceptable": self.is_acceptable,
            "blur_score": round(self.blur_score, 2),
            "is_sharp": self.is_sharp,
            "mean_intensity": round(self.mean_intensity, 2),
            "illumination_status": self.illumination_status,
            "fov_ratio": round(self.fov_ratio, 4),
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "details": self.details,
        }


def to_numpy_rgb(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Standardize an input image into a uint8 RGB numpy array.

    Args:
        image_input: File path, PIL Image, or NumPy array (RGB or BGR).

    Returns:
        np.ndarray: uint8 image in RGB color format.
    """
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Fundus image not found at path: {path_str}")
        bgr = cv2.imread(path_str)
        if bgr is None:
            raise ValueError(f"Failed to decode image from path: {path_str}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    if isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return arr

    raise TypeError(f"Unsupported image type: {type(image_input)}")


def compute_blur_score(image_rgb: np.ndarray) -> float:
    """
    Estimate sharpness using the variance of the Laplacian operator on the green channel.
    The green channel in retinal imaging provides the highest anatomical contrast
    for blood vessels and retinal landmarks.

    Args:
        image_rgb: NumPy array of the fundus image in RGB format.

    Returns:
        float: Laplacian variance score. Higher is sharper, lower is blurrier.
    """
    if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
        # Use Green channel (index 1) for fundus imaging
        green_channel = image_rgb[:, :, 1]
    else:
        green_channel = image_rgb

    laplacian = cv2.Laplacian(green_channel, cv2.CV_64F)
    score = float(laplacian.var())
    return score


def compute_illumination_metrics(image_rgb: np.ndarray, threshold_black: int = 15) -> dict[str, float]:
    """
    Compute illumination metrics specifically within the illuminated fundus region,
    ignoring the surrounding black camera borders.

    Args:
        image_rgb: RGB fundus image.
        threshold_black: Pixel value threshold below which pixels are treated as black background.

    Returns:
        dict: Mean intensity, standard deviation (contrast), and saturation ratio within the fundus.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold_black

    if not np.any(mask):
        return {
            "mean_intensity": 0.0,
            "contrast_std": 0.0,
            "overexposed_ratio": 0.0,
            "underexposed_ratio": 1.0,
        }

    fundus_pixels = gray[mask]
    mean_val = float(np.mean(fundus_pixels))
    std_val = float(np.std(fundus_pixels))
    overexposed = float(np.mean(fundus_pixels >= 245))
    underexposed = float(np.mean(fundus_pixels <= 25))

    return {
        "mean_intensity": mean_val,
        "contrast_std": std_val,
        "overexposed_ratio": overexposed,
        "underexposed_ratio": underexposed,
    }


def detect_and_crop_fov(
    image_rgb: np.ndarray, threshold: int = 15, padding_ratio: float = 0.02
) -> Tuple[np.ndarray, dict[str, int]]:
    """
    Detect the circular retinal Field of View (FOV) and crop uninformative
    black borders surrounding the retina.

    Args:
        image_rgb: Input RGB image.
        threshold: Threshold intensity to separate the illuminated retina from background.
        padding_ratio: Small relative buffer around the detected bounding box.

    Returns:
        Tuple[np.ndarray, dict]: Cropped RGB image and bounding box coordinates (ymin, ymax, xmin, xmax).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold

    if not np.any(mask):
        # Fallback if image is entirely black
        h, w = gray.shape
        return image_rgb, {"ymin": 0, "ymax": h, "xmin": 0, "xmax": w}

    y_indices, x_indices = np.where(mask)
    ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
    xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))

    h, w = gray.shape
    pad_y = int((ymax - ymin) * padding_ratio)
    pad_x = int((xmax - xmin) * padding_ratio)

    ymin = max(0, ymin - pad_y)
    ymax = min(h, ymax + pad_y)
    xmin = max(0, xmin - pad_x)
    xmax = min(w, xmax + pad_x)

    cropped = image_rgb[ymin:ymax, xmin:xmax]
    bbox = {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}
    return cropped, bbox


def apply_circular_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply a soft circular mask fitting the image dimensions, zeroing out corners
    outside the retinal disk to eliminate sensor artifacts or camera frame edges.

    Args:
        image_rgb: Input RGB image (usually cropped to the retina).

    Returns:
        np.ndarray: Circularly masked RGB image.
    """
    h, w = image_rgb.shape[:2]
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) / 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255,), thickness=-1)

    result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return result


def ben_graham_preprocessing(image_rgb: np.ndarray, sigma_x: int = 10) -> np.ndarray:
    """
    Apply Ben Graham's local contrast enhancement (popularized in Kaggle Diabetic Retinopathy).
    Subtracts local average color to mitigate lighting variation across different fundus cameras.

    Args:
        image_rgb: Input RGB image.
        sigma_x: Gaussian blur kernel scale.

    Returns:
        np.ndarray: Contrast-normalized RGB image.
    """
    # cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigma_x), -4, 128)
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigma_x)
    enhanced = cv2.addWeighted(image_rgb, 4, blurred, -4, 128)
    return enhanced


def assess_fundus_quality(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    blur_threshold: float = 80.0,
    min_mean_intensity: float = 30.0,
    max_mean_intensity: float = 220.0,
    min_fov_ratio: float = 0.20,
) -> QualityReport:
    """
    Perform a comprehensive quality screening on a retinal fundus photograph.

    Evaluates:
    - Focus/Sharpness (Laplacian variance on green channel)
    - Illumination status (underexposed vs overexposed)
    - Field-of-View ratio (sufficient retinal tissue visible)

    Args:
        image_input: Image path, PIL Image, or numpy array.
        blur_threshold: Threshold below which the image is deemed blurry.
        min_mean_intensity: Minimum average pixel value within the fundus.
        max_mean_intensity: Maximum average pixel value within the fundus.
        min_fov_ratio: Minimum ratio of retinal tissue to full frame area.

    Returns:
        QualityReport: Structured report detailing diagnostic suitability.
    """
    image_rgb = to_numpy_rgb(image_input)
    h, w = image_rgb.shape[:2]
    total_pixels = h * w

    # Calculate blur score
    blur_score = compute_blur_score(image_rgb)
    is_sharp = bool(blur_score >= blur_threshold)

    # Illumination metrics
    illum_metrics = compute_illumination_metrics(image_rgb)
    mean_intensity = illum_metrics["mean_intensity"]

    # FOV ratio
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    retinal_pixels = int(np.count_nonzero(gray > 15))
    fov_ratio = float(retinal_pixels / total_pixels) if total_pixels > 0 else 0.0

    warnings: list[str] = []
    recommendations: list[str] = []

    if not is_sharp:
        warnings.append(
            f"Image appears blurred (sharpness score {blur_score:.1f} < threshold {blur_threshold:.1f})."
        )
        recommendations.append("Re-capture photograph with proper patient fixation and camera autofocus.")

    if mean_intensity < min_mean_intensity:
        illum_status = "underexposed"
        warnings.append(
            f"Image is underexposed/dark (mean intensity {mean_intensity:.1f} < {min_mean_intensity:.1f})."
        )
        recommendations.append("Increase camera flash/illumination or ensure pupil dilation.")
    elif mean_intensity > max_mean_intensity:
        illum_status = "overexposed"
        warnings.append(
            f"Image is overexposed/glare (mean intensity {mean_intensity:.1f} > {max_mean_intensity:.1f})."
        )
        recommendations.append("Reduce camera flash intensity to prevent macular washout.")
    else:
        illum_status = "adequate"

    if fov_ratio < min_fov_ratio:
        warnings.append(
            f"Retinal Field of View is abnormally small (ratio {fov_ratio:.2f} < {min_fov_ratio:.2f})."
        )
        recommendations.append("Align the camera closer to the patient's eye to fill the aperture.")

    is_acceptable = is_sharp and (illum_status == "adequate") and (fov_ratio >= min_fov_ratio)

    return QualityReport(
        is_acceptable=is_acceptable,
        blur_score=blur_score,
        is_sharp=is_sharp,
        mean_intensity=mean_intensity,
        illumination_status=illum_status,
        fov_ratio=fov_ratio,
        warnings=warnings,
        recommendations=recommendations,
        details={
            "height": h,
            "width": w,
            "contrast_std": round(illum_metrics["contrast_std"], 2),
            "overexposed_ratio": round(illum_metrics["overexposed_ratio"], 4),
            "underexposed_ratio": round(illum_metrics["underexposed_ratio"], 4),
        },
    )
