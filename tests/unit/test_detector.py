"""
Unit tests for Detection Service (Phase 4).
Runs completely offline with mocked YOLO and MockDetector (zero GPU, zero network).
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from app.core.config import settings
from app.schemas.domain import BoundingBoxSchema
from app.services.detector import (
    DetectedObject,
    InvalidImageError,
    MockDetector,
    ModelWeightsNotFoundError,
    YOLOAerialDetector,
    get_optimal_torch_device,
)


@pytest.fixture
def synthetic_aerial_image(tmp_path: Path) -> Path:
    """Creates a local synthetic 300x200 RGB test image."""
    img_path = tmp_path / "test_aerial.jpg"
    img = np.full((200, 300, 3), (120, 140, 100), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


# ==========================================
# 1. Safety Terminology Tests
# ==========================================

def test_detected_object_sanitizes_prohibited_terms():
    """Safety Rule 1: 'confirmed survivor' is never output by the detection service."""
    bbox = BoundingBoxSchema(x1=10, y1=10, x2=50, y2=50)

    obj1 = DetectedObject.create_safe("confirmed survivor", 0.92, bbox)
    assert obj1.class_name == "person"
    assert "confirmed" not in obj1.class_name

    obj2 = DetectedObject.create_safe("survivor confirmed", 0.88, bbox)
    assert obj2.class_name == "person"

    obj3 = DetectedObject.create_safe("alive person", 0.75, bbox)
    assert obj3.class_name == "person"

    obj4 = DetectedObject.create_safe("person", 0.89, bbox)
    assert obj4.class_name == "person"


# ==========================================
# 2. MockDetector Offline Tests
# ==========================================

def test_mock_detector_prediction(synthetic_aerial_image: Path):
    detector = MockDetector()
    result = detector.predict(synthetic_aerial_image)

    assert result.total_detections >= 1
    assert result.person_count >= 1
    assert result.image_width == 300
    assert result.image_height == 200

    obj = result.objects[0]
    assert obj.class_name == "person"
    assert 0.0 <= obj.confidence <= 1.0
    assert obj.bbox.x1 < obj.bbox.x2
    assert obj.bbox.y1 < obj.bbox.y2


def test_mock_detector_configurable_confidence_threshold(synthetic_aerial_image: Path):
    mock_objs = [
        DetectedObject.create_safe("person", 0.90, BoundingBoxSchema(x1=10, y1=10, x2=30, y2=50)),
        DetectedObject.create_safe("person", 0.40, BoundingBoxSchema(x1=50, y1=50, x2=70, y2=90)),
        DetectedObject.create_safe("vehicle", 0.20, BoundingBoxSchema(x1=80, y1=80, x2=100, y2=120)),
    ]
    detector = MockDetector(mock_objects=mock_objs)

    # Low threshold (0.20) keeps all 3
    res_low = detector.predict(synthetic_aerial_image, conf_threshold=0.20)
    assert res_low.total_detections == 3

    # High threshold (0.80) keeps only the 0.90 detection
    res_high = detector.predict(synthetic_aerial_image, conf_threshold=0.80)
    assert res_high.total_detections == 1
    assert res_high.objects[0].confidence == 0.90


def test_mock_detector_save_annotated_image(synthetic_aerial_image: Path, tmp_path: Path):
    detector = MockDetector()
    annotated_out = tmp_path / "annotated" / "out.jpg"

    result = detector.predict(synthetic_aerial_image, save_annotated_path=annotated_out)

    assert result.annotated_image_path == str(annotated_out)
    assert annotated_out.exists()
    assert annotated_out.stat().st_size > 0


# ==========================================
# 3. Error Handling Tests (Missing Weights & Bad Images)
# ==========================================

def test_yolo_detector_missing_weights_error(synthetic_aerial_image: Path, tmp_path: Path):
    """Verifies that missing weights file raises clear error instead of downloading from web."""
    non_existent_weights = tmp_path / "weights" / "missing_yolov8n.pt"
    detector = YOLOAerialDetector(weights_path=non_existent_weights)

    with pytest.raises(ModelWeightsNotFoundError) as exc_info:
        detector.predict(synthetic_aerial_image)

    assert "Model weights not found" in str(exc_info.value)
    assert "Automatic downloading of weights is disabled" in str(exc_info.value)


def test_detector_invalid_image_path(tmp_path: Path):
    detector = MockDetector()
    with pytest.raises(InvalidImageError) as exc_info:
        detector.predict(tmp_path / "non_existent.jpg")
    assert "does not exist" in str(exc_info.value)


def test_detector_empty_image_array():
    detector = MockDetector()
    with pytest.raises(InvalidImageError) as exc_info:
        detector.predict(np.array([]))
    assert "empty" in str(exc_info.value)


# ==========================================
# 4. Mocked Ultralytics YOLO Adapter Tests
# ==========================================

def test_yolo_adapter_mocked_inference(synthetic_aerial_image: Path, tmp_path: Path):
    """
    Mocks the Ultralytics YOLO class to verify YOLOAerialDetector logic
    without requiring real model weights, GPU, or internet access.
    """
    fake_weights = tmp_path / "fake_weights.pt"
    fake_weights.write_text("dummy model weights")

    # Mock ultralytics YOLO result
    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=lambda: 0.88)]
    mock_box.cls = [MagicMock(item=lambda: 0)]
    mock_box.xyxy = [[15.0, 20.0, 55.0, 110.0]]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person", 1: "bicycle"}

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.predict.return_value = [mock_result]

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        detector = YOLOAerialDetector(weights_path=fake_weights, default_conf_threshold=0.30)
        res = detector.predict(synthetic_aerial_image)

        assert res.total_detections == 1
        assert res.person_count == 1
        assert res.objects[0].class_name == "person"
        assert res.objects[0].confidence == 0.88
        assert res.objects[0].bbox.x1 == 15.0
        assert res.objects[0].bbox.y1 == 20.0
        assert res.objects[0].bbox.x2 == 55.0
        assert res.objects[0].bbox.y2 == 110.0
        mock_yolo_instance.predict.assert_called_once()


def test_yolo_device_selection_cuda_and_cpu():
    """Verifies CUDA and CPU device selection logic and fallback safety."""
    with patch("torch.cuda.is_available", return_value=True):
        assert get_optimal_torch_device(None) == 0
        assert get_optimal_torch_device("cuda") == 0
        assert get_optimal_torch_device("0") == 0
        assert get_optimal_torch_device("cpu") == "cpu"

    with patch("torch.cuda.is_available", return_value=False):
        assert get_optimal_torch_device(None) == "cpu"
        assert get_optimal_torch_device("cuda") == "cpu"
        assert get_optimal_torch_device("0") == "cpu"
        assert get_optimal_torch_device("cpu") == "cpu"


def test_yolo_detector_person_only_filtering(synthetic_aerial_image: Path, tmp_path: Path):
    """Verifies that person_only=True filters predictions to class 0 = person."""
    fake_weights = tmp_path / "fake_weights.pt"
    fake_weights.write_text("dummy model weights")

    box_person = MagicMock()
    box_person.conf = [MagicMock(item=lambda: 0.90)]
    box_person.cls = [MagicMock(item=lambda: 0)]
    box_person.xyxy = [[10.0, 10.0, 50.0, 100.0]]

    box_vehicle = MagicMock()
    box_vehicle.conf = [MagicMock(item=lambda: 0.85)]
    box_vehicle.cls = [MagicMock(item=lambda: 1)]
    box_vehicle.xyxy = [[60.0, 60.0, 120.0, 150.0]]

    box_other = MagicMock()
    box_other.conf = [MagicMock(item=lambda: 0.70)]
    box_other.cls = [MagicMock(item=lambda: 2)]
    box_other.xyxy = [[130.0, 130.0, 180.0, 180.0]]

    mock_result = MagicMock()
    mock_result.boxes = [box_person, box_vehicle, box_other]
    mock_result.names = {0: "person", 1: "vehicle", 2: "other"}

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.predict.return_value = [mock_result]

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        # 1. person_only=True: only class 0 is retained
        detector_person_only = YOLOAerialDetector(
            weights_path=fake_weights,
            default_conf_threshold=0.35,
            person_only=True,
        )
        res_person_only = detector_person_only.predict(synthetic_aerial_image)
        assert res_person_only.total_detections == 1
        assert res_person_only.person_count == 1
        assert res_person_only.objects[0].class_name == "person"

        # 2. person_only=False: all classes are retained
        detector_all = YOLOAerialDetector(
            weights_path=fake_weights,
            default_conf_threshold=0.35,
            person_only=False,
        )
        res_all = detector_all.predict(synthetic_aerial_image)
        assert res_all.total_detections == 3
        assert res_all.person_count == 1
        classes = [o.class_name for o in res_all.objects]
        assert "person" in classes
        assert "vehicle" in classes
        assert "other" in classes


def test_yolo_detector_configurable_confidence_threshold(synthetic_aerial_image: Path, tmp_path: Path):
    """Verifies that configurable confidence threshold correctly filters low-confidence detections."""
    fake_weights = tmp_path / "fake_weights.pt"
    fake_weights.write_text("dummy model weights")

    box1 = MagicMock()
    box1.conf = [MagicMock(item=lambda: 0.80)]
    box1.cls = [MagicMock(item=lambda: 0)]
    box1.xyxy = [[10.0, 10.0, 50.0, 100.0]]

    box2 = MagicMock()
    box2.conf = [MagicMock(item=lambda: 0.30)]
    box2.cls = [MagicMock(item=lambda: 0)]
    box2.xyxy = [[60.0, 60.0, 120.0, 150.0]]

    mock_result = MagicMock()
    mock_result.boxes = [box1, box2]
    mock_result.names = {0: "person"}

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.predict.return_value = [mock_result]

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        # Default 0.35 filters out box2 (0.30)
        detector = YOLOAerialDetector(weights_path=fake_weights, default_conf_threshold=0.35)
        res = detector.predict(synthetic_aerial_image)
        assert res.total_detections == 1
        assert res.objects[0].confidence == 0.80

        # Overridden lower threshold 0.25 includes both
        res_low = detector.predict(synthetic_aerial_image, conf_threshold=0.25)
        assert res_low.total_detections == 2


def test_trained_visdrone_model_loading_and_inference(synthetic_aerial_image: Path):
    """Verifies that the trained VisDrone model loads and executes inference on the configured device."""
    trained_weights = settings.resolved_yolo_weights_path
    if not trained_weights.exists():
        pytest.skip(f"Trained weights not found at {trained_weights}")

    detector = YOLOAerialDetector(
        weights_path=trained_weights,
        default_conf_threshold=0.35,
        person_only=True,
    )
    assert detector.weights_path.exists()
    assert detector.person_only is True

    # Run inference on synthetic test image
    result = detector.predict(synthetic_aerial_image)
    assert isinstance(result.total_detections, int)
    assert isinstance(result.person_count, int)
    assert result.image_width == 300
    assert result.image_height == 200
    assert result.model_name == "best"

