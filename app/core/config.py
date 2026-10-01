"""
Application Configuration and Environment Settings.
Includes defensive operational caps and security boundaries.
"""
import os
from pathlib import Path
from typing import Optional, Set
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "AI-Enabled UAV Disaster Assessment"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    BASE_DIR: Path = BASE_DIR
    
    # Environment & Database
    ENV: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    DATABASE_URL: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/disaster_uav.db")
    )
    
    # Storage Paths
    UPLOAD_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    )
    WEIGHTS_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("WEIGHTS_DIR", str(BASE_DIR / "weights")))
    )
    
    # Defensive Operational Limits
    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))  # 50 MB
    )
    MAX_VIDEO_DURATION_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "600.0"))  # 10 minutes
    )
    MAX_FRAMES_PER_VIDEO: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FRAMES_PER_VIDEO", "300"))  # Max frames extracted per video
    )
    MIN_SAMPLING_INTERVAL_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("MIN_SAMPLING_INTERVAL_SECONDS", "0.1"))
    )
    ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    
    # Privacy & Logging Security
    ENABLE_LOG_MASKING: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_LOG_MASKING", "true").lower() in ("true", "1", "yes")
    )
    LOG_LEVEL: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Detector Defaults
    USE_MOCK_DETECTOR: bool = Field(
        default_factory=lambda: os.getenv("USE_MOCK_DETECTOR", "true").lower() in ("true", "1", "yes")
    )
    YOLO_MODEL_NAME: str = Field(
        default_factory=lambda: os.getenv("YOLO_MODEL_NAME", "best.pt")
    )
    YOLO_MODEL_PATH: str = Field(
        default_factory=lambda: os.getenv(
            "YOLO_MODEL_PATH",
            "runs/detect/runs/detect/visdrone_yolov8n_cuda_v1/weights/best.pt",
        )
    )
    YOLO_DEVICE: Optional[str] = Field(
        default_factory=lambda: os.getenv("YOLO_DEVICE", None)
    )
    YOLO_PERSON_ONLY: bool = Field(
        default_factory=lambda: os.getenv("YOLO_PERSON_ONLY", "true").lower() in ("true", "1", "yes")
    )
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.35"))
    )

    # RescueNet Semantic Segmentation Defaults
    ENABLE_RESCUENET_SEGMENTATION: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_RESCUENET_SEGMENTATION", "false").lower() in ("true", "1", "yes")
    )
    RESCUENET_SEGMENTATION_MODEL_PATH: str = Field(
        default_factory=lambda: os.getenv(
            "RESCUENET_SEGMENTATION_MODEL_PATH",
            "runs/rescuenet_pilot/best_model.pt",
        )
    )
    RESCUENET_SEGMENTATION_DEVICE: Optional[str] = Field(
        default_factory=lambda: os.getenv("RESCUENET_SEGMENTATION_DEVICE", None)
    )

    @property
    def resolved_rescuenet_segmentation_weights_path(self) -> Path:
        """Resolves the RescueNet segmentation model weights path relative to BASE_DIR."""
        raw_path = self.RESCUENET_SEGMENTATION_MODEL_PATH
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.BASE_DIR / path
        return path

    @property
    def resolved_yolo_weights_path(self) -> Path:
        """
        Resolves the YOLO model weights path relative to BASE_DIR if not absolute.
        Prefers YOLO_MODEL_PATH, falling back to WEIGHTS_DIR / YOLO_MODEL_NAME.
        """
        raw_path = self.YOLO_MODEL_PATH
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.BASE_DIR / path
        if path.exists():
            return path
        return self.WEIGHTS_DIR / self.YOLO_MODEL_NAME
    
    # Safety & Telemetry Defaults
    DEFAULT_SIMULATED_ALTITUDE_M: float = 45.0
    DEFAULT_GIMBAL_PITCH_DEG: float = -90.0  # Nadir
    DEFAULT_CAMERA_FOV_DEG: float = 84.0
    DEFAULT_SIMULATED_BASE_LAT: float = 34.0522
    DEFAULT_SIMULATED_BASE_LON: float = -118.2437


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
