"""
Oculi: Diabetic Retinopathy Screening Research Prototype
========================================================

A modular computer vision pipeline for automated retinal fundus image
quality verification, 5-grade diabetic retinopathy severity classification
using EfficientNet-B0, visual explainability (Grad-CAM), and clinical triage.
"""

__version__ = "0.1.0"
__author__ = "Oculi Research Team"

from oculi.models.backbone import OculiEfficientNetB0, build_model
from oculi.models.predictor import DRPredictor, PredictionResult, TriageEngine
from oculi.preprocessing.quality import QualityReport, assess_fundus_quality

__all__ = [
    "__version__",
    "OculiEfficientNetB0",
    "build_model",
    "DRPredictor",
    "PredictionResult",
    "TriageEngine",
    "QualityReport",
    "assess_fundus_quality",
]
