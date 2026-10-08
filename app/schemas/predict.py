"""Pydantic DTO schemas for API request and response data models."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class Top3PredictionSchema(BaseModel):
    """Schema representing an individual breed rank item in Top-3 predictions."""

    class_id: int = Field(description="Breed class integer index (0..5)")
    breed_name: str = Field(description="Breed canonical name (e.g. 'Gir')")
    display_name: str = Field(description="Formatted display name (e.g. 'Gir Cattle')")
    animal_type: str = Field(description="Animal classification ('cattle' or 'buffalo')")
    confidence: float = Field(description="Prediction probability confidence [0.0, 1.0]")


class InferenceTimeSchema(BaseModel):
    """Schema representing latency breakdown in milliseconds."""

    detection_ms: float = Field(description="YOLO animal detection latency in milliseconds")
    classification_ms: float = Field(description="EfficientNet-B0 breed classification latency in milliseconds")
    gradcam_ms: float = Field(description="Grad-CAM explainability latency in milliseconds")
    total_ms: float = Field(description="End-to-end total request latency in milliseconds")


class GradCAMResponseSchema(BaseModel):
    """Schema representing Grad-CAM visual explainability outputs."""

    target_class_id: int = Field(description="Target class ID evaluated")
    display_name: str = Field(description="Target breed display name")
    overlay_base64: str = Field(description="Base64 data URL encoded PNG overlay image")
    heatmap_base64: Optional[str] = Field(default=None, description="Base64 data URL encoded PNG heatmap image")


class PredictResponseSchema(BaseModel):
    """Schema representing full prediction output for POST /api/predict."""

    animal_type: str = Field(description="Detected species ('cattle', 'buffalo', or 'unknown')")
    animal_confidence: float = Field(description="YOLO animal detection confidence score")
    bounding_box: Tuple[int, int, int, int] = Field(description="Pixel bounding box tuple (xmin, ymin, xmax, ymax)")
    predicted_breed: str = Field(description="Top-1 predicted breed name")
    breed_confidence: float = Field(description="Top-1 breed prediction confidence score")
    top_3_predictions: List[Top3PredictionSchema] = Field(description="Top-3 ranked breed predictions")
    prediction_status: str = Field(
        description="Status string ('success', 'no_animal_detected', 'invalid_image', 'low_detection_confidence', 'low_classification_confidence', 'multiple_animals_detected', 'model_failure')"
    )
    inference_time: InferenceTimeSchema = Field(description="Latency breakdown metrics")
    model_versions: Dict[str, str] = Field(description="Model version metadata")
    gradcam_output: Optional[GradCAMResponseSchema] = Field(default=None, description="Grad-CAM explainability data")

    # Phase 24 Expanded Fields (Backwards Compatible & Fully Populated)
    model_version: Optional[str] = Field(default=None, description="Active model version identifier")
    species: Optional[str] = Field(default=None, description="Species classification ('cattle' or 'buffalo')")
    confidence: Optional[float] = Field(default=None, description="Top-1 prediction confidence score")
    top_3: Optional[List[Top3PredictionSchema]] = Field(default=None, description="Top-3 ranked breed predictions list")
    latency: Optional[float] = Field(default=None, description="Total inference latency in milliseconds")
    status: Optional[str] = Field(default=None, description="Canonical prediction status string")
    request_id: Optional[str] = Field(default=None, description="Unique trace request ID")
    breed: Optional[str] = Field(default=None, description="Top-1 predicted breed name")
    top_predictions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Top candidate predictions list")
    processing_time: Optional[float] = Field(default=None, description="Inference latency in ms")


class BreedInfoSchema(BaseModel):
    """Schema representing breed metadata detail."""

    breed_name: str = Field(description="Canonical breed name")
    display_name: str = Field(description="Formatted display name")
    animal_type: str = Field(description="Animal classification ('cattle' or 'buffalo')")
    origin_region: str = Field(description="Origin region in India")
    description: str = Field(description="Breed characteristics summary")


class BreedsListResponseSchema(BaseModel):
    """Schema representing response for GET /api/breeds."""

    total_breeds: int = Field(description="Total supported breed count")
    breeds: List[BreedInfoSchema] = Field(description="Supported breed catalog list")


class ModelInfoResponseSchema(BaseModel):
    """Schema representing response for GET /api/model-info."""

    system_name: str = Field(description="Project system name")
    pipeline_version: str = Field(description="Overall AI pipeline version")
    supported_species: List[str] = Field(description="Supported animal species")
    supported_breeds_count: int = Field(description="Count of supported breeds")
    checkpoints_status: Dict[str, bool] = Field(description="Trained model checkpoints presence status")
    environment_info: Dict[str, Any] = Field(description="PyTorch and compute hardware environment info")
