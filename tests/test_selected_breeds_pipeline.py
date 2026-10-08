"""Unit tests verifying the selected-breed high-accuracy (8-class) model and API integration."""

import cv2
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from ml.pipeline.inference_pipeline import BreedRecognitionPipeline

client = TestClient(app)


def test_selected_breed_model_files_exist():
    """Verify all model checkpoint and configuration artifacts exist."""
    model_dir = Path("models/selected_breeds")
    assert (model_dir / "best_selected_breed_model.pth").exists()
    assert (model_dir / "class_mapping.json").exists()
    assert (model_dir / "model_config.json").exists()
    assert (model_dir / "training_config.json").exists()
    assert (model_dir / "normalization_config.json").exists()


def test_selected_breed_pipeline_initialization():
    """Verify that BreedRecognitionPipeline loads the 8 selected classes properly."""
    pipeline = BreedRecognitionPipeline(
        efficientnet_model_path="models/selected_breeds/best_selected_breed_model.pth",
        class_mapping_path="models/selected_breeds/class_mapping.json",
    )
    assert len(pipeline.breed_predictor.class_mapping) == 8
    expected_breeds = {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    actual_breeds = {info["breed_name"] for info in pipeline.breed_predictor.class_mapping.values()}
    assert actual_breeds == expected_breeds


def test_selected_breed_pipeline_dummy_inference():
    """Verify end-to-end inference and Grad-CAM generation for the 8-class champion model."""
    pipeline = BreedRecognitionPipeline(
        efficientnet_model_path="models/selected_breeds/best_selected_breed_model.pth",
        class_mapping_path="models/selected_breeds/class_mapping.json",
    )
    # Generate random test image
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    res = pipeline.predict(img, generate_gradcam=True)

    assert res.prediction_status in ["success", "no_animal_detected", "low_classification_confidence"]
    assert len(res.top_3_predictions) == 3
    assert res.predicted_breed in {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}


def test_api_breeds_selected_version():
    """Verify GET /api/breeds?version=selected_breeds_v1 returns the 8 curated breeds."""
    response = client.get("/api/breeds?version=selected_breeds_v1")
    assert response.status_code == 200
    data = response.json()

    assert data["total_breeds"] == 8
    assert len(data["breeds"]) == 8

    breed_names = {b["breed_name"] for b in data["breeds"]}
    assert breed_names == {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}

    # Verify cattle and buffalo split
    cattle = [b for b in data["breeds"] if b["animal_type"] == "cattle"]
    buffalo = [b for b in data["breeds"] if b["animal_type"] == "buffalo"]
    assert len(cattle) == 5
    assert len(buffalo) == 3


def test_api_predict_selected_breeds_default():
    """Verify POST /api/predict defaults to selected_breeds_v4 with all required fields."""
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)

    files = {"file": ("test_animal.png", buffer.tobytes(), "image/png")}
    response = client.post("/api/predict?generate_gradcam=true", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] in ["selected_breeds_swin_v2", "selected_breeds_v4"]
    assert "breed" in data
    assert "species" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    assert "processing_time" in data
    assert data["processing_time"] > 0
    assert data["predicted_breed"] in {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}


def test_selected_breed_v4_model_files_exist():
    """Verify all selected_breeds_v4 model checkpoint and configuration artifacts exist."""
    model_dir = Path("models/selected_breeds/selected_breeds_v4")
    assert (model_dir / "best_selected_breed_model.pth").exists()
    assert (model_dir / "class_mapping.json").exists()
    assert (model_dir / "model_config.json").exists()
    assert (model_dir / "training_history.json").exists()


def test_selected_breed_v4_pipeline_initialization():
    """Verify that BreedRecognitionPipeline loads the 8 selected classes for v4."""
    pipeline = BreedRecognitionPipeline(
        efficientnet_model_path="models/selected_breeds/selected_breeds_v4/best_selected_breed_model.pth",
        class_mapping_path="models/selected_breeds/selected_breeds_v4/class_mapping.json",
    )
    assert len(pipeline.breed_predictor.class_mapping) == 8
    expected_breeds = {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    actual_breeds = {info["breed_name"] for info in pipeline.breed_predictor.class_mapping.values()}
    assert actual_breeds == expected_breeds


def test_api_predict_selected_breeds_v4():
    """Verify POST /api/predict works explicitly with model_version=selected_breeds_v4."""
    img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)

    files = {"file": ("test_animal_v4.png", buffer.tobytes(), "image/png")}
    response = client.post("/api/predict?generate_gradcam=true&model_version=selected_breeds_v4", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] == "selected_breeds_v4"
    assert "breed" in data
    assert "species" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    assert data["processing_time"] > 0
    assert data["predicted_breed"] in {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    assert data["species"] in {"cattle", "buffalo"}


def test_selected_breed_v2_model_files_exist():
    """Verify all selected_breeds_v2 model checkpoint and configuration artifacts exist."""
    model_dir = Path("models/selected_breeds/selected_breeds_v2")
    assert (model_dir / "best_selected_breed_model.pth").exists()
    assert (model_dir / "final_selected_breed_model.pth").exists()
    assert (model_dir / "class_mapping.json").exists()
    assert (model_dir / "model_config.json").exists()
    assert (model_dir / "training_config.json").exists()
    assert (model_dir / "normalization_config.json").exists()


def test_selected_breed_v2_pipeline_initialization():
    """Verify that BreedRecognitionPipeline loads the 8 selected classes for v2."""
    pipeline = BreedRecognitionPipeline(
        efficientnet_model_path="models/selected_breeds/selected_breeds_v2/best_selected_breed_model.pth",
        class_mapping_path="models/selected_breeds/selected_breeds_v2/class_mapping.json",
    )
    assert len(pipeline.breed_predictor.class_mapping) == 8
    expected_breeds = {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    actual_breeds = {info["breed_name"] for info in pipeline.breed_predictor.class_mapping.values()}
    assert actual_breeds == expected_breeds


def test_api_predict_selected_breeds_v2():
    """Verify POST /api/predict works with model_version=selected_breeds_v2."""
    img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)

    files = {"file": ("test_animal_v2.png", buffer.tobytes(), "image/png")}
    response = client.post("/api/predict?generate_gradcam=true&model_version=selected_breeds_v2", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] == "selected_breeds_v2"
    assert "breed" in data
    assert "species" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    assert data["processing_time"] > 0
    assert data["predicted_breed"] in {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    assert data["species"] in {"cattle", "buffalo"}


def test_api_breeds_selected_v2_version():
    """Verify GET /api/breeds?version=selected_breeds_v2 returns the 8 curated breeds."""
    response = client.get("/api/breeds?version=selected_breeds_v2")
    assert response.status_code == 200
    data = response.json()

    assert data["total_breeds"] == 8
    assert len(data["breeds"]) == 8
    breed_names = {b["breed_name"] for b in data["breeds"]}
    assert breed_names == {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}


def test_selected_breed_swin_synth_v3_model_files_exist():
    """Verify all selected_breeds_swin_synth_v3 model checkpoint and configuration artifacts exist."""
    model_dir = Path("models/selected_breeds/swin_synth_v3")
    assert (model_dir / "best_swin_synth_model.pth").exists()
    assert (model_dir / "class_mapping.json").exists()
    assert (model_dir / "model_metadata.json").exists()
    assert (model_dir / "preprocessing.json").exists()
    assert (model_dir / "training_config.json").exists()


def test_selected_breed_swin_synth_v3_pipeline_initialization():
    """Verify that BreedRecognitionPipeline loads the 8 selected classes for Swin-T Synthetic V3."""
    pipeline = BreedRecognitionPipeline(
        efficientnet_model_path="models/selected_breeds/swin_synth_v3/best_swin_synth_model.pth",
        class_mapping_path="models/selected_breeds/swin_synth_v3/class_mapping.json",
    )
    assert len(pipeline.breed_predictor.class_mapping) == 8
    expected_breeds = {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    actual_breeds = {info["breed_name"] for info in pipeline.breed_predictor.class_mapping.values()}
    assert actual_breeds == expected_breeds


def test_api_predict_selected_breeds_swin_synth_v3():
    """Verify POST /api/predict works explicitly with model_version=selected_breeds_swin_synth_v3."""
    img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)

    files = {"file": ("test_animal_swin.png", buffer.tobytes(), "image/png")}
    response = client.post("/api/predict?generate_gradcam=true&model_version=selected_breeds_swin_synth_v3", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] == "selected_breeds_swin_synth_v3"
    assert "breed" in data
    assert "species" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    assert data["processing_time"] > 0
    assert data["predicted_breed"] in {"Dangi", "Gaolao", "Gir", "Ongole", "Sahiwal", "Manda", "Surti", "Toda"}
    assert data["species"] in {"cattle", "buffalo"}


