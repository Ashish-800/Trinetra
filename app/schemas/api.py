"""
API Schemas for Phase 10 FastAPI Endpoints.
Defines validated request payloads, partial updates, and API metadata models.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.domain import BoundingBoxSchema
from app.schemas.location import DroneTelemetryMetadata


class FrontendIncidentSummary(BaseModel):
    """Frontend-friendly incident metadata."""
    id: int
    title: str
    disaster_type: str
    location_name: Optional[str] = None
    is_active: bool = True


class FrontendMediaSummary(BaseModel):
    """Frontend-friendly media asset metadata."""
    id: int
    media_type: str = "IMAGE"
    file_path: str
    file_size_bytes: int = 0
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    frames_processed: int = 1


class FrontendDetectionItem(BaseModel):
    """Frontend-friendly detection item with safe terminology and simulated location."""
    detection_id: Optional[int] = None
    track_id: Optional[int] = None
    label: str = Field(
        default="potential person",
        description="Always 'potential person' or 'person detected', never 'confirmed survivor'"
    )
    confidence: float
    bbox: BoundingBoxSchema
    frame_number: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_source: str = Field(
        default="simulated",
        description="Explicitly indicates coordinates are simulated UAV projection estimates"
    )
    uncertainty_radius_m: Optional[float] = None
    review_status: str = "PENDING_REVIEW"


class FrontendDeviceInfo(BaseModel):
    """Hardware and model telemetry for command-center diagnostic display."""
    cuda_available: bool
    inference_device: str
    gpu_name: Optional[str] = None
    detector_model: str = "visdrone_yolov8n_cuda_v1/best.pt"
    segmentation_model: str = "rescuenet_pilot/best_model.pt"


class AnalysisRequest(BaseModel):
    """Request payload to initiate automated assessment on registered media."""
    media_asset_id: int = Field(..., description="Database ID of registered MediaAsset to analyze", gt=0)
    telemetry: Optional[DroneTelemetryMetadata] = Field(
        None,
        description="Optional UAV telemetry for ray-casting location projection. If omitted, default simulated baseline is used."
    )
    sampling_interval_seconds: float = Field(
        default=1.0,
        description="Frame extraction sampling interval in seconds for video media",
        gt=0.0,
        le=60.0
    )
    detection_confidence_threshold: Optional[float] = Field(
        None,
        description="Optional confidence threshold filter for object detection",
        ge=0.05,
        le=1.0
    )
    rescuenet_mask_path: Optional[str] = Field(
        None,
        description="Optional path to a RescueNet disaster segmentation mask to extract environmental damage context"
    )
    enable_segmentation: Optional[bool] = Field(
        None,
        description="Optional flag to run real-time RescueNet semantic segmentation inference on the input image"
    )


class ResourceUpdate(BaseModel):
    """Schema for updating an existing simulated emergency resource record."""
    name: Optional[str] = Field(None, min_length=2, max_length=120)
    resource_type: Optional[str] = Field(None, min_length=2, max_length=50)
    total_capacity: Optional[int] = Field(None, gt=0)
    available_capacity: Optional[int] = Field(None, ge=0)
    simulated_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    simulated_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    is_available: Optional[bool] = None

    @field_validator("available_capacity")
    @classmethod
    def validate_available_capacity(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None and "total_capacity" in info.data and info.data["total_capacity"] is not None:
            if v > info.data["total_capacity"]:
                raise ValueError(f"available_capacity ({v}) cannot exceed total_capacity ({info.data['total_capacity']})")
        return v


class HealthResponse(BaseModel):
    """System health and operational status response."""
    status: str = "healthy"
    app_name: str
    version: str
    environment: str
    safety_constitution: str = "Strictly enforced: Advisory only, zero autonomous physical actuation"
    detector_mode: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Standardized API error response body."""
    detail: str
    error_code: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
