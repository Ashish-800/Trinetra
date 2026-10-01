"""
Detection Service and Ultralytics YOLO Adapter.
Provides a decoupled interface for aerial object detection,
enforcing safety terminology and offline predictability.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from pydantic import BaseModel, Field
from app.schemas.domain import BoundingBoxSchema


class DetectorError(Exception):
    """Base exception for detection service errors."""
    pass


class ModelWeightsNotFoundError(DetectorError):
    """Raised when the specified model weights file does not exist locally."""
    pass


class InvalidImageError(DetectorError):
    """Raised when an image path does not exist or image data is corrupted."""
    pass


class DetectedObject(BaseModel):
    """Normalized detected entity."""
    class_name: str = Field(..., description="Detected label, e.g. 'person' or 'vehicle'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBoxSchema

    @classmethod
    def create_safe(
        cls,
        raw_class_name: Optional[str] = None,
        confidence: float = 0.0,
        bbox: Optional[BoundingBoxSchema] = None,
        *,
        class_name: Optional[str] = None,
        **kwargs,
    ) -> "DetectedObject":
        """
        Factory method enforcing Safety Rule 1:
        Never label any detection as 'confirmed survivor', 'alive', or 'deceased'.
        Maps person-like detections strictly to 'person' or 'potential survivor'.
        """
        target_name = class_name if class_name is not None else (raw_class_name if raw_class_name is not None else kwargs.get("class_name", ""))
        normalized = target_name.strip().lower()
        prohibited = ["confirmed survivor", "survivor confirmed", "alive", "deceased"]
        for p in prohibited:
            if p in normalized:
                # Sanitize to safe terminology
                normalized = "person"
                break
        
        return cls(class_name=normalized, confidence=confidence, bbox=bbox)


class DetectionResult(BaseModel):
    """Aggregated output from a single image inference pass."""
    objects: List[DetectedObject]
    annotated_image_path: Optional[str] = None
    total_detections: int
    person_count: int
    image_width: int
    image_height: int
    model_name: str


class BaseDetector(ABC):
    """Abstract interface decoupling the application from specific model versions."""

    @abstractmethod
    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        save_annotated_path: Optional[Union[str, Path]] = None,
    ) -> DetectionResult:
        """Runs object detection on the provided image."""
        pass


import logging

logger = logging.getLogger(__name__)


def get_optimal_torch_device(preferred_device: Optional[Union[int, str]] = None) -> Union[int, str]:
    """
    Resolves the optimal execution device for YOLO inference.
    Prefers CUDA (device 0) when available, falling back safely to CPU.
    """
    if preferred_device is not None:
        dev_str = str(preferred_device).strip().lower()
        if dev_str in ("cuda", "gpu", "0"):
            try:
                import torch
                if torch.cuda.is_available():
                    return 0
                logger.warning(
                    "CUDA requested (%s) but torch.cuda.is_available() is False; falling back to CPU",
                    preferred_device,
                )
                return "cpu"
            except Exception:
                return "cpu"
        elif dev_str == "cpu":
            return "cpu"
        return preferred_device

    # Auto-detection: prefer CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


class YOLOAerialDetector(BaseDetector):
    """
    Ultralytics YOLO adapter for aerial imagery inference.
    Does not download weights automatically; requires an explicit local weights file.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        default_conf_threshold: float = 0.35,
        device: Optional[Union[int, str]] = None,
        target_classes: Optional[List[str]] = None,
        model_weights: Optional[Union[str, Path]] = None,
        person_only: bool = False,
    ):
        resolved_weights = weights_path if weights_path is not None else model_weights
        if resolved_weights is None:
            raise ValueError("Either weights_path or model_weights must be provided.")
        self.weights_path = Path(resolved_weights)
        self.default_conf_threshold = default_conf_threshold
        self.device = get_optimal_torch_device(device)
        self.target_classes = [c.lower() for c in target_classes] if target_classes else None
        self.person_only = person_only
        self._model = None

    def _ensure_model_loaded(self):
        """Loads model weights lazily. Raises error if weights are missing locally."""
        if self._model is not None:
            return

        if not self.weights_path.exists() or not self.weights_path.is_file():
            raise ModelWeightsNotFoundError(
                f"Model weights not found at: '{self.weights_path}'. "
                "Automatic downloading of weights is disabled by safety policy. "
                "Please place a valid YOLO weights file (.pt) in the designated path."
            )

        try:
            from ultralytics import YOLO  # Lazy import
            self._model = YOLO(str(self.weights_path))
            if self.device is not None and hasattr(self._model, "to"):
                self._model.to(self.device)
        except Exception as e:
            raise DetectorError(f"Failed to initialize Ultralytics YOLO model from '{self.weights_path}': {e}") from e

    def _prepare_image(self, image_input: Union[str, Path, np.ndarray]) -> tuple[np.ndarray, int, int]:
        """Validates and loads the input image into a BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists() or not path.is_file():
                raise InvalidImageError(f"Image file does not exist: '{path}'")
            img = cv2.imread(str(path))
            if img is None:
                raise InvalidImageError(f"Could not decode image at '{path}'. File may be corrupt.")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0 or len(image_input.shape) not in (2, 3):
                raise InvalidImageError("Input numpy array is empty or has invalid dimensions.")
            img = image_input.copy()
            if len(img.shape) == 2:  # Grayscale to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")

        height, width = img.shape[:2]
        return img, width, height

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        save_annotated_path: Optional[Union[str, Path]] = None,
    ) -> DetectionResult:
        """Executes inference on image_input using YOLO."""
        self._ensure_model_loaded()
        img, width, height = self._prepare_image(image_input)
        threshold = conf_threshold if conf_threshold is not None else self.default_conf_threshold

        try:
            results = self._model.predict(
                source=img,
                conf=threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise DetectorError(f"Inference execution failed: {e}") from e

        detected_objects: List[DetectedObject] = []
        person_count = 0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                names = result.names or {}
                for box in boxes:
                    conf = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])
                    if conf < threshold:
                        continue

                    cls_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
                    raw_name = names.get(cls_id, str(cls_id))

                    xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                    # Clip to image bounds
                    x1, y1 = max(0.0, x1), max(0.0, y1)
                    x2, y2 = min(float(width), max(x1, x2)), min(float(height), max(y1, y2))

                    # Safety terminology enforcement
                    safe_obj = DetectedObject.create_safe(
                        raw_class_name=raw_name,
                        confidence=conf,
                        bbox=BoundingBoxSchema(x1=x1, y1=y1, x2=x2, y2=y2),
                    )

                    # Person-only filtering for primary search workflow (VisDrone class 0 = person)
                    if self.person_only and cls_id != 0 and safe_obj.class_name not in ("person", "potential survivor"):
                        continue

                    # Optional target class filtering
                    if self.target_classes and safe_obj.class_name not in self.target_classes:
                        continue

                    if safe_obj.class_name in ("person", "potential survivor"):
                        person_count += 1

                    detected_objects.append(safe_obj)

        annotated_path_str = None
        if save_annotated_path:
            out_path = Path(save_annotated_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_img = self._annotate_image(img, detected_objects)
            cv2.imwrite(str(out_path), annotated_img)
            annotated_path_str = str(out_path)

        return DetectionResult(
            objects=detected_objects,
            annotated_image_path=annotated_path_str,
            total_detections=len(detected_objects),
            person_count=person_count,
            image_width=width,
            image_height=height,
            model_name=self.weights_path.stem,
        )

    def _annotate_image(self, img: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
        """Draws bounding boxes and labels on an image copy."""
        annotated = img.copy()
        for obj in objects:
            x1, y1 = int(obj.bbox.x1), int(obj.bbox.y1)
            x2, y2 = int(obj.bbox.x2), int(obj.bbox.y2)

            color = (0, 0, 255) if obj.class_name in ("person", "potential survivor") else (255, 165, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated


class MockDetector(BaseDetector):
    """
    Deterministic mock detector for unit testing.
    Runs completely offline with zero GPU and zero network requirements.
    """

    def __init__(
        self,
        mock_objects: Optional[List[DetectedObject]] = None,
        model_name: str = "mock_yolo_v8",
        canned_detections: Optional[List[DetectedObject]] = None,
        **kwargs,
    ):
        self.mock_objects = canned_detections if canned_detections is not None else mock_objects
        self.model_name = model_name

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        save_annotated_path: Optional[Union[str, Path]] = None,
    ) -> DetectionResult:
        """Generates deterministic detections without real model inference."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists() or not path.is_file():
                raise InvalidImageError(f"Image file does not exist: '{path}'")
            img = cv2.imread(str(path))
            if img is None:
                raise InvalidImageError(f"Could not decode image at '{path}'")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise InvalidImageError("Input numpy array is empty.")
            img = image_input.copy()
        else:
            raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")

        height, width = img.shape[:2]
        threshold = conf_threshold if conf_threshold is not None else 0.35

        if self.mock_objects is not None:
            raw_objects = self.mock_objects
        else:
            # Deterministic default mock detection: 1 person in the center
            cx, cy = width / 2.0, height / 2.0
            raw_objects = [
                DetectedObject.create_safe(
                    raw_class_name="person",
                    confidence=0.85,
                    bbox=BoundingBoxSchema(
                        x1=max(0.0, cx - 20.0),
                        y1=max(0.0, cy - 40.0),
                        x2=min(float(width), cx + 20.0),
                        y2=min(float(height), cy + 40.0),
                    ),
                )
            ]

        # Filter by threshold and enforce safety rules
        filtered_objects = [
            DetectedObject.create_safe(
                raw_class_name=obj.class_name,
                confidence=obj.confidence,
                bbox=obj.bbox,
            )
            for obj in raw_objects
            if obj.confidence >= threshold
        ]

        person_count = sum(1 for o in filtered_objects if o.class_name in ("person", "potential survivor"))

        annotated_path_str = None
        if save_annotated_path:
            out_path = Path(save_annotated_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = img.copy()
            for obj in filtered_objects:
                cv2.rectangle(
                    annotated,
                    (int(obj.bbox.x1), int(obj.bbox.y1)),
                    (int(obj.bbox.x2), int(obj.bbox.y2)),
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(str(out_path), annotated)
            annotated_path_str = str(out_path)

        return DetectionResult(
            objects=filtered_objects,
            annotated_image_path=annotated_path_str,
            total_detections=len(filtered_objects),
            person_count=person_count,
            image_width=width,
            image_height=height,
            model_name=self.model_name,
        )
