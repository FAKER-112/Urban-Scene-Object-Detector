import os
import sys
import json
import pytest
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.utils.config_parser import load_config
from src.pipeline.predict_pipeline import run_yolo_prediction 

# test predict pipeline
@pytest.fixture
def sample_image(tmp_path):
    """Provide a small sample image for testing."""
    import cv2
    import numpy as np
    img_path = tmp_path / "test_image.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def dummy_model(tmp_path):
    """Create or locate a YOLO model path for testing."""
    # Normally you'd use a small YOLOv8n model. Adjust path accordingly.
    model_path = "artifacts/models/runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        pytest.skip("YOLOv8 model not available for test.")
    return model_path


def test_run_yolo_prediction_image(dummy_model, sample_image, tmp_path):
    """Test image prediction output and metrics structure."""
    output_dir = tmp_path / "runs"
    metrics = run_yolo_prediction(
        model_path=dummy_model,
        input_source=sample_image,
        output_dir=str(output_dir),
        conf_threshold=0.25
    )

    # Validate return type and structure
    assert isinstance(metrics, dict)
    for key in [
        "model_path", "input_source", "is_video",
        "output_dir", "total_time_sec", "total_detections",
        "avg_confidence", "detection_summary"
    ]:
        assert key in metrics

    # Check metrics file saved
    metrics_path = os.path.join(metrics["output_dir"], "metrics.json")
    assert os.path.exists(metrics_path)

    # Validate JSON content
    with open(metrics_path, "r") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["model_path"] == dummy_model


def test_run_yolo_prediction_no_detections(dummy_model, sample_image, tmp_path):
    """Ensure pipeline runs even if no detections are made."""
    metrics = run_yolo_prediction(
        model_path=dummy_model,
        input_source=sample_image,
        output_dir=str(tmp_path)
    )
    assert metrics["total_detections"] >= 0
    
