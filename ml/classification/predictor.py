"""Breed Classification Inference Predictor module.

Executes EfficientNet-B0 inference on animal image crops and returns Top-1 prediction,
confidence score, and Top-3 predictions.
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from ml.common.breed_registry import BreedRegistry
from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor
from ml.classification.efficientnet import BreedClassifier

# Default class mapping dictionary
DEFAULT_CLASS_MAPPING: Dict[int, Dict[str, str]] = {
    0: {"breed_name": "Gir", "animal_type": "cattle", "display_name": "Gir Cattle"},
    1: {"breed_name": "Ongole", "animal_type": "cattle", "display_name": "Ongole Cattle"},
    2: {"breed_name": "Sahiwal", "animal_type": "cattle", "display_name": "Sahiwal Cattle"},
    3: {"breed_name": "Jaffarabadi", "animal_type": "buffalo", "display_name": "Jaffarabadi Buffalo"},
    4: {"breed_name": "Murrah", "animal_type": "buffalo", "display_name": "Murrah Buffalo"},
    5: {"breed_name": "Surti", "animal_type": "buffalo", "display_name": "Surti Buffalo"},
}


class BreedPredictionItem(BaseModel):
    """Individual breed prediction item for Top-K rankings."""

    class_id: int = Field(description="Class integer index (0..5)")
    breed_name: str = Field(description="Breed name string (e.g. 'Gir')")
    display_name: str = Field(description="Formatted display name (e.g. 'Gir Cattle')")
    animal_type: str = Field(description="Animal classification ('cattle' or 'buffalo')")
    confidence: float = Field(description="Probability confidence score [0.0, 1.0]")


class PredictionResult(BaseModel):
    """Pydantic model representing complete breed prediction output."""

    predicted_breed: str = Field(description="Top-1 predicted breed name")
    display_name: str = Field(description="Top-1 formatted display name")
    animal_type: str = Field(description="Predicted animal classification")
    confidence: float = Field(description="Top-1 prediction confidence score")
    top_3_predictions: List[BreedPredictionItem] = Field(description="Top-3 ranked breed predictions")

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        return {
            "predicted_breed": self.predicted_breed,
            "display_name": self.display_name,
            "animal_type": self.animal_type,
            "confidence": round(self.confidence, 4),
            "top_3_predictions": [item.model_dump() for item in self.top_3_predictions],
        }


class BreedPredictor:
    """Predictor engine running EfficientNet-B0 inference on animal image crops."""

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        class_mapping_path: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
    ):
        """Initialize BreedPredictor.

        Args:
            model_path: Path to trained PyTorch weights (.pth). If None, checks models/efficientnet_best.pth.
            class_mapping_path: Path to class_mapping.json file.
            device: Compute device ('cpu' or 'cuda:0').
        """
        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # Load class mapping
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.num_classes = len(self.class_mapping)

        # Initialize preprocessor
        self.preprocessor = OpenCVPreprocessor(target_size=(224, 224))

        # Determine architecture (b0, b4, or swin_t)
        arch = "b0"
        if model_path and "swin" in str(model_path).lower():
            arch = "swin_t"
        elif model_path and ("v4" in str(model_path).lower() or "b4" in str(model_path).lower()):
            arch = "b4"

        # Load weights if checkpoint exists
        if model_path is None:
            if self.num_classes == 82:
                default_weights = Path("models/efficientnet_b0_82_breeds_best.pth")
                if default_weights.exists():
                    model_path = default_weights
            else:
                default_weights = Path("models/efficientnet_best.pth")
                if default_weights.exists():
                    model_path = default_weights

        loaded_state_dict = None
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            loaded_state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            if any(k.startswith("head.") or "mlp" in k for k in loaded_state_dict.keys()):
                arch = "swin_t"
            if any(k.startswith("features.") or k.startswith("classifier.") or k.startswith("head.") or k.startswith("norm.") for k in loaded_state_dict.keys()):
                loaded_state_dict = {f"backbone.{k}": v for k, v in loaded_state_dict.items()}
            # Check feature projection dim (1792 in B4, 1280 in B0)
            for k, v in loaded_state_dict.items():
                if "classifier.1.weight" in k:
                    if v.shape[1] == 1792:
                        arch = "b4"
                    break

        # Instantiate model architecture
        self.model = BreedClassifier(num_classes=self.num_classes, pretrained=False, arch=arch)
        if loaded_state_dict is not None:
            self.model.load_state_dict(loaded_state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_class_mapping(
        self, class_mapping_path: Optional[Union[Path, str]]
    ) -> Dict[int, Dict[str, str]]:
        """Load class mapping dictionary from JSON file or default map."""
        raw_json = None
        if class_mapping_path and Path(class_mapping_path).exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
        elif Path("configs/class_mapping.json").exists():
            with open("configs/class_mapping.json", "r", encoding="utf-8") as f:
                raw_json = json.load(f)

        if raw_json is not None:
            # Check if list of class dicts
            if isinstance(raw_json, list):
                mapping = {}
                for item in raw_json:
                    cls_idx = int(item.get("class_id", len(mapping)))
                    sp = item.get("animal_type", item.get("species", "cattle"))
                    mapping[cls_idx] = {
                        "breed_name": item.get("breed_name", f"Breed_{cls_idx}"),
                        "animal_type": sp,
                        "display_name": item.get("display_name", f"{item.get('breed_name', '')} {sp.capitalize()}"),
                    }
                return mapping

            # Check if rich format like models/class_names.json
            if "idx_to_breed_name" in raw_json and "idx_to_species" in raw_json:
                mapping = {}
                for k, bname in raw_json["idx_to_breed_name"].items():
                    sp = raw_json["idx_to_species"].get(k, "cattle")
                    mapping[int(k)] = {
                        "breed_name": bname,
                        "animal_type": sp,
                        "display_name": f"{bname} {sp.capitalize()}",
                    }
                return mapping

            # Check if dict of strings or dict of dicts
            mapping = {}
            for k, v in raw_json.items():
                if not k.isdigit():
                    continue
                cls_idx = int(k)
                if isinstance(v, dict):
                    mapping[cls_idx] = {
                        "breed_name": v.get("breed_name", f"Breed_{cls_idx}"),
                        "animal_type": v.get("animal_type", "cattle"),
                        "display_name": v.get("display_name", f"{v.get('breed_name', '')}"),
                    }
                elif isinstance(v, str):
                    sp = "buffalo" if cls_idx < 23 else "cattle"
                    mapping[cls_idx] = {
                        "breed_name": v,
                        "animal_type": sp,
                        "display_name": f"{v} {sp.capitalize()}",
                    }
            if mapping:
                return mapping

        return DEFAULT_CLASS_MAPPING

    def predict(
        self,
        source: Union[Path, str, bytes, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        top_k: int = 3,
    ) -> PredictionResult:
        """Run breed classification inference on image or cropped ROI.

        Args:
            source: Image path, string path, bytes, or numpy array.
            bbox: Optional bounding box tuple (xmin, ymin, xmax, ymax) to crop prior to classification.
            top_k: Number of top predictions to return (default 3).

        Returns:
            PredictionResult Pydantic DTO.
        """
        # Preprocess input image to tensor (3, 224, 224)
        tensor_img = self.preprocessor.preprocess_to_tensor(source, bbox=bbox)
        batch_tensor = tensor_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        # Retrieve Top-K probabilities and indices
        k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k=k, largest=True, sorted=True)

        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top_3_items: List[BreedPredictionItem] = []
        for idx, prob in zip(top_indices, top_probs):
            cls_id = int(idx)
            info = self.class_mapping.get(
                cls_id,
                {
                    "breed_name": f"Breed_{cls_id}",
                    "animal_type": "cattle",
                    "display_name": f"Breed {cls_id}",
                },
            )

            item = BreedPredictionItem(
                class_id=cls_id,
                breed_name=info["breed_name"],
                display_name=info["display_name"],
                animal_type=info["animal_type"],
                confidence=float(prob),
            )
            top_3_items.append(item)

        top_1 = top_3_items[0]
        return PredictionResult(
            predicted_breed=top_1.breed_name,
            display_name=top_1.display_name,
            animal_type=top_1.animal_type,
            confidence=top_1.confidence,
            top_3_predictions=top_3_items,
        )
