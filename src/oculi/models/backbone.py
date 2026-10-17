"""
EfficientNet-B0 Backbone Architecture for Diabetic Retinopathy Classification
============================================================================

Provides the convolutional backbone based on torchvision's EfficientNet-B0,
configured with a custom classification head for 5-level diabetic retinopathy
severity staging (Grades 0 to 4).

NOTE: The model uses ImageNet-pretrained weights for feature extraction.
It is an architectural research prototype and is NOT fine-tuned for clinical
diagnosis until trained on validated retinal datasets.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)


class OculiEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 customized for 5-grade Diabetic Retinopathy staging.

    Classes:
        0: No DR
        1: Mild NPDR
        2: Moderate NPDR
        3: Severe NPDR
        4: Proliferative DR
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Load torchvision EfficientNet-B0 backbone
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.base_model = efficientnet_b0(weights=weights)
                logger.info("Loaded EfficientNet-B0 with ImageNet-1K pretrained weights.")
            except Exception as exc:
                logger.warning(
                    f"Could not download ImageNet weights ({exc}). Initializing random weights."
                )
                self.base_model = efficientnet_b0(weights=None)
        else:
            self.base_model = efficientnet_b0(weights=None)

        # Extract features block and classification head dimensions
        # efficientnet_b0.classifier[1] is nn.Linear(in_features=1280, out_features=1000)
        in_features = self.base_model.classifier[1].in_features

        # Replace standard 1000-class classifier with 5-class DR head
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    @property
    def features(self) -> nn.Sequential:
        """Access the convolutional feature extractor blocks."""
        return self.base_model.features

    @property
    def target_cam_layer(self) -> nn.Module:
        """Returns the final convolutional block used for Grad-CAM activation mapping."""
        # The final stage in efficientnet_b0 features is features[8]
        return self.base_model.features[-1]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps before pooling.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Feature map of shape (B, 1280, H', W').
        """
        return self.base_model.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding unnormalized class logits.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Unnormalized logits of shape (B, num_classes).
        """
        return self.base_model(x)


def build_model(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.2,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> OculiEfficientNetB0:
    """
    Factory function to instantiate OculiEfficientNetB0 and optionally load trained weights.

    Args:
        num_classes: Number of target classes (default 5 for DR).
        pretrained: If True, uses ImageNet pre-training for features.
        dropout: Dropout rate in classifier head.
        checkpoint_path: Optional path to a trained .pth checkpoint file.
        device: Torch device to map weights to.

    Returns:
        OculiEfficientNetB0: Instantiated and ready model.
    """
    model = OculiEfficientNetB0(num_classes=num_classes, pretrained=pretrained, dropout=dropout)

    if checkpoint_path is not None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded trained checkpoint from: {checkpoint_path}")

    if device is not None:
        model.to(device)

    return model
