"""Complete AI Inference Pipeline module for Indian Cattle and Buffalo Breed Recognition.

Connects Image -> YOLO Animal Detection -> OpenCV Crop -> EfficientNet-B0 Breed Classification -> Grad-CAM Explainability.
Supports configurable inference backend: 'pytorch' or 'onnx'.
"""

import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor
from ml.detection.yolo_detector import YOLOAnimalDetector, DetectionResult
from ml.classification.predictor import BreedPredictor, PredictionResult
from ml.explainability.gradcam import EfficientNetGradCAM, GradCAMResult


class InferenceResult(BaseModel):
    """Pydantic model representing complete end-to-end AI inference output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    animal_type: str = Field(description="Detected animal classification ('cattle' or 'buffalo')")
    animal_confidence: float = Field(description="YOLO animal detection confidence score")
    bounding_box: Tuple[int, int, int, int] = Field(
        description="Pixel bounding box tuple (xmin, ymin, xmax, ymax)"
    )
    predicted_breed: str = Field(description="Top-1 predicted breed name")
    breed_confidence: float = Field(description="Top-1 breed prediction confidence score")
    top_3_predictions: List[Dict[str, Any]] = Field(
        description="Top-3 ranked breed predictions list"
    )
    prediction_status: str = Field(
        description="Status string ('success', 'no_animal_detected', 'invalid_image', 'low_detection_confidence', 'low_classification_confidence')"
    )
    inference_time: Dict[str, float] = Field(
        description="Detailed latency breakdown in milliseconds (detection_ms, classification_ms, gradcam_ms, total_ms)"
    )
    model_versions: Dict[str, str] = Field(
        description="Model architecture and weight version metadata"
    )
    gradcam_output: Optional[Dict[str, Any]] = Field(
        default=None, description="Grad-CAM explainability heatmap and overlay dictionary"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert InferenceResult to dictionary."""
        res = self.model_dump()
        res["animal_confidence"] = round(self.animal_confidence, 4)
        res["breed_confidence"] = round(self.breed_confidence, 4)
        return res


class BreedRecognitionPipeline:
    """Unified end-to-end AI inference service."""

    def __init__(
        self,
        yolo_model_path: Optional[Union[Path, str]] = None,
        efficientnet_model_path: Optional[Union[Path, str]] = None,
        class_mapping_path: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
        backend: str = "pytorch",
    ):
        """Initialize BreedRecognitionPipeline.

        Args:
            yolo_model_path: Optional path to trained YOLO weights (.pt).
            efficientnet_model_path: Optional path to trained EfficientNet-B0 weights (.pth).
            class_mapping_path: Optional path to class_mapping.json.
            device: Compute device ('cpu' or 'cuda:0').
            backend: Inference backend ('pytorch' or 'onnx').
        """
        self.backend = backend.lower()
        self.preprocessor = OpenCVPreprocessor(target_size=(224, 224))
        self.yolo_detector = YOLOAnimalDetector(model_path=yolo_model_path)
        self.breed_predictor = BreedPredictor(
            model_path=efficientnet_model_path,
            class_mapping_path=class_mapping_path,
            device=device,
        )
        self.gradcam_engine = EfficientNetGradCAM(
            model=self.breed_predictor.model,
            model_path=efficientnet_model_path,
            class_mapping_path=class_mapping_path,
            device=device,
        )

        self.onnx_predictor = None
        if self.backend == "onnx":
            onnx_path = Path("models/onnx/efficientnet_b0.onnx")
            if onnx_path.exists():
                from ml.export.onnx_predictor import ONNXBreedPredictor
                self.onnx_predictor = ONNXBreedPredictor(
                    onnx_model_path=str(onnx_path),
                    class_mapping=self.breed_predictor.class_mapping,
                )

        self.model_versions = {
            "yolo_detector": "YOLOv8n-AnimalDetection-v1.0",
            "breed_classifier": f"EfficientNet-B0-BreedRecognition-v1.0 ({self.backend.upper()})",
            "explainability": "Grad-CAM-v1.0",
            "backend": self.backend,
        }

    def predict(
        self,
        source: Union[Path, str, bytes, np.ndarray],
        generate_gradcam: bool = True,
        detection_conf_threshold: float = 0.25,
        classification_conf_threshold: float = 0.20,
    ) -> InferenceResult:
        """Execute end-to-end breed recognition inference.

        Args:
            source: Image file path, bytes, or numpy array.
            generate_gradcam: Whether to compute Grad-CAM explainability heatmap overlay.
            detection_conf_threshold: Minimum detection confidence score.
            classification_conf_threshold: Minimum classification confidence score.

        Returns:
            InferenceResult Pydantic DTO.
        """
        t_start = time.perf_counter()
        detection_ms = 0.0
        classification_ms = 0.0
        gradcam_ms = 0.0

        # 1. Image Loading and Validation
        try:
            raw_img = self.preprocessor.load_image(source)
            orig_rgb = self.preprocessor.ensure_rgb(raw_img)
            h, w = orig_rgb.shape[:2]
        except Exception:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return InferenceResult(
                animal_type="unknown",
                animal_confidence=0.0,
                bounding_box=(0, 0, 0, 0),
                predicted_breed="unknown",
                breed_confidence=0.0,
                top_3_predictions=[],
                prediction_status="invalid_image",
                inference_time={
                    "detection_ms": 0.0,
                    "classification_ms": 0.0,
                    "gradcam_ms": 0.0,
                    "total_ms": round(total_ms, 2),
                },
                model_versions=self.model_versions,
                gradcam_output=None,
            )

        # 2. YOLO Animal Detection
        t_det_start = time.perf_counter()
        detections: List[DetectionResult] = []
        try:
            detections = self.yolo_detector.detect_animals(
                source=orig_rgb, conf_threshold=detection_conf_threshold
            )
        except Exception:
            pass
        detection_ms = (time.perf_counter() - t_det_start) * 1000.0

        if not detections:
            primary_bbox = (0, 0, w, h)
            animal_type = "unknown"
            animal_conf = 0.0
            status = "no_animal_detected"
        else:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            primary_det = detections[0]
            primary_bbox = primary_det.bbox
            animal_type = primary_det.class_name
            animal_conf = primary_det.confidence

            if len(detections) > 1:
                status = "multiple_animals_detected"
            elif animal_conf < detection_conf_threshold:
                status = "low_detection_confidence"
            else:
                status = "success"

        # 3. Crop Animal ROI & Preprocess for EfficientNet-B0
        crops = self.yolo_detector.crop_animals(
            image_rgb=orig_rgb,
            detections=[
                DetectionResult(
                    class_id=0 if animal_type == "cattle" else 1,
                    class_name=animal_type if animal_type != "unknown" else "cattle",
                    confidence=animal_conf,
                    bbox=primary_bbox,
                )
            ],
            pad_percent=0.05,
        )
        crop_rgb = crops[0]["crop"] if crops else orig_rgb

        # 4. EfficientNet-B0 Breed Classification (PyTorch vs ONNX)
        t_cls_start = time.perf_counter()
        if self.backend == "onnx" and self.onnx_predictor is not None:
            # ONNX Runtime inference
            tensor_img = self.breed_predictor.preprocessor.preprocess_to_tensor(crop_rgb)
            onnx_res = self.onnx_predictor.predict(tensor_img.unsqueeze(0))

            pred_breed = onnx_res["predicted_breed"]
            confidence = onnx_res["confidence"]
            top_3_list = onnx_res["top_3_predictions"]
        else:
            # PyTorch inference
            pred_res: PredictionResult = self.breed_predictor.predict(crop_rgb, top_k=3)
            pred_breed = pred_res.predicted_breed
            confidence = pred_res.confidence
            top_3_list = [item.model_dump() for item in pred_res.top_3_predictions]

        classification_ms = (time.perf_counter() - t_cls_start) * 1000.0

        if status == "success" and confidence < classification_conf_threshold:
            status = "low_classification_confidence"

        # 5. Grad-CAM Explainability Generation
        gradcam_dict = None
        if generate_gradcam and status in ["success", "multiple_animals_detected", "low_classification_confidence", "no_animal_detected"]:
            t_grad_start = time.perf_counter()
            try:
                target_cls_id = top_3_list[0]["class_id"] if top_3_list else 0
                gradcam_res: GradCAMResult = self.gradcam_engine.generate_gradcam(
                    source=crop_rgb, target_class=target_cls_id
                )
                gradcam_dict = {
                    "display_name": gradcam_res.display_name,
                    "target_class_id": gradcam_res.target_class_id,
                    "heatmap": gradcam_res.heatmap,
                    "overlay_image": gradcam_res.overlay_image,
                }
            except Exception:
                pass
            gradcam_ms = (time.perf_counter() - t_grad_start) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return InferenceResult(
            animal_type=animal_type if animal_type != "unknown" else "cattle",
            animal_confidence=round(animal_conf, 4),
            bounding_box=primary_bbox,
            predicted_breed=pred_breed,
            breed_confidence=round(confidence, 4),
            top_3_predictions=top_3_list,
            prediction_status=status,
            inference_time={
                "detection_ms": round(detection_ms, 2),
                "classification_ms": round(classification_ms, 2),
                "gradcam_ms": round(gradcam_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            model_versions=self.model_versions,
            gradcam_output=gradcam_dict,
        )
