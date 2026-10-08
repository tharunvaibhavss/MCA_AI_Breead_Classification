"""Grad-CAM (Gradient-weighted Class Activation Mapping) Engine for EfficientNet-B0.

Provides visual explainability by highlighting image regions contributing to breed predictions.
Model parameters and weights remain completely untouched.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, ConfigDict

from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor
from ml.classification.efficientnet import BreedClassifier
from ml.classification.predictor import BreedPredictor, PredictionResult


class GradCAMResult(BaseModel):
    """Pydantic DTO holding Grad-CAM explainability outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predicted_breed: str = Field(description="Predicted breed name")
    display_name: str = Field(description="Formatted display name")
    animal_type: str = Field(description="Predicted animal classification")
    confidence: float = Field(description="Prediction confidence score")
    target_class_id: int = Field(description="Target class ID evaluated")
    original_image: np.ndarray = Field(description="Original RGB image array (H, W, 3)")
    heatmap: np.ndarray = Field(description="Normalized 2D heatmap array (H, W) in [0.0, 1.0]")
    overlay_image: np.ndarray = Field(description="Blended RGB overlay image array (H, W, 3)")


class EfficientNetGradCAM:
    """Grad-CAM generator registered non-destructively on EfficientNet-B0."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[Union[Path, str]] = None,
        class_mapping_path: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
    ):
        """Initialize EfficientNetGradCAM.

        Args:
            model: Optional instantiated BreedClassifier model.
            model_path: Optional path to saved PyTorch checkpoint.
            class_mapping_path: Optional path to class_mapping.json.
            device: Compute device ('cpu' or 'cuda:0').
        """
        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.preprocessor = OpenCVPreprocessor(target_size=(224, 224))
        self.predictor = BreedPredictor(
            model_path=model_path,
            class_mapping_path=class_mapping_path,
            device=str(self.device),
        )

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self.predictor.model

        self.model.eval()

        # Locate final convolutional feature layer of EfficientNet-B0
        self.target_layer = self._find_target_layer()

        # Hook storage variables
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Locate final conv layer of EfficientNet-B0 backbone (features[-1])."""
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            return self.model.backbone.features[-1]
        # Fallback to last child module if architecture differs
        children = list(self.model.children())
        return children[-2] if len(children) > 1 else children[-1]

    def _register_hooks(self) -> None:
        """Register forward and backward execution hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(
        self,
        source: Union[Path, str, bytes, np.ndarray],
        target_class: Optional[int] = None,
        alpha: float = 0.5,
    ) -> GradCAMResult:
        """Generate Grad-CAM heatmap and blended overlay for input image.

        Args:
            source: Image file path, bytes, or RGB numpy array.
            target_class: Optional target class ID. If None, uses Top-1 predicted class.
            alpha: Heatmap blend ratio [0.0, 1.0]. Default 0.5.

        Returns:
            GradCAMResult containing original image, heatmap, and overlay array.
        """
        # Load and validate original RGB image
        raw_img = self.preprocessor.load_image(source)
        orig_rgb = self.preprocessor.ensure_rgb(raw_img)
        h, w = orig_rgb.shape[:2]

        # Convert image to input tensor
        tensor_img = self.preprocessor.preprocess_to_tensor(orig_rgb)
        input_batch = tensor_img.unsqueeze(0).to(self.device)
        input_batch.requires_grad = True

        # Forward pass
        self.model.zero_grad()
        logits = self.model(input_batch)
        probs = F.softmax(logits, dim=1).squeeze(0)

        if target_class is None:
            target_class = int(torch.argmax(probs).item())

        target_prob = float(probs[target_class].item())

        # Backward pass on target class score
        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture Grad-CAM activations or gradients.")

        # Handle channels-last (B, H, W, C) from Vision Transformers / Swin vs CNNs (B, C, H, W)
        act = self.activations
        grad = self.gradients
        if act.dim() == 4 and act.shape[3] > act.shape[1] and act.shape[3] > act.shape[2]:
            act = act.permute(0, 3, 1, 2)
            grad = grad.permute(0, 3, 1, 2)

        # Compute channel-wise mean gradient weights
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1).squeeze(0)

        # Apply ReLU activation to filter negative importance
        cam = F.relu(cam).cpu().numpy()

        # Normalize CAM heatmap to range [0.0, 1.0]
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match original image dimensions (H, W)
        heatmap_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

        # Convert normalized heatmap to RGB JET colormap overlay
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend original RGB image with JET heatmap
        overlay = cv2.addWeighted(orig_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

        # Get breed metadata info
        pred_res = self.predictor.predict(orig_rgb, top_k=1)

        return GradCAMResult(
            predicted_breed=pred_res.predicted_breed,
            display_name=pred_res.display_name,
            animal_type=pred_res.animal_type,
            confidence=round(target_prob, 4),
            target_class_id=target_class,
            original_image=orig_rgb,
            heatmap=heatmap_resized,
            overlay_image=overlay,
        )
