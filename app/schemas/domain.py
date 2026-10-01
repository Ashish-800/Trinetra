"""
Validated Pydantic Domain Schemas for Disaster UAV System.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LocationSource(str, Enum):
    SIMULATED = "simulated"
    GPS_EXIF = "gps_exif"
    MANUAL_INPUT = "manual_input"
    UNKNOWN = "unknown"


class ReviewStatusEnum(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    DISMISSED = "DISMISSED"


class PriorityLevelEnum(str, Enum):
    CRITICAL_REVIEW = "CRITICAL_REVIEW"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BoundingBoxSchema(BaseModel):
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")

    @field_validator("x2")
    @classmethod
    def validate_x2(cls, v: float, info) -> float:
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError(f"x2 ({v}) cannot be less than x1 ({info.data['x1']})")
        return v

    @field_validator("y2")
    @classmethod
    def validate_y2(cls, v: float, info) -> float:
        if "y1" in info.data and v < info.data["y1"]:
            raise ValueError(f"y2 ({v}) cannot be less than y1 ({info.data['y1']})")
        return v


# 1. Incident Schemas
class IncidentCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=150)
    disaster_type: str = Field(default="GENERAL")
    location_name: str = Field(..., min_length=2, max_length=200)
    notes: Optional[str] = None


class IncidentRead(BaseModel):
    id: int
    title: str
    disaster_type: str
    location_name: str
    notes: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# 2. MediaAsset Schemas
class MediaAssetCreate(BaseModel):
    incident_id: int
    file_path: str = Field(..., min_length=1)
    media_type: str = Field(..., pattern="^(IMAGE|VIDEO)$")
    mime_type: str = Field(..., min_length=3)
    file_size_bytes: int = Field(..., ge=0)
    width_px: Optional[int] = Field(None, gt=0)
    height_px: Optional[int] = Field(None, gt=0)
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    frame_count: Optional[int] = Field(None, ge=0)


class MediaAssetRead(MediaAssetCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# 3. Track Schemas
class TrackCreate(BaseModel):
    incident_id: int
    track_label: str = Field(..., min_length=1, max_length=80)
    is_active: bool = True
    start_time: Optional[datetime] = None


class TrackRead(BaseModel):
    id: int
    incident_id: int
    track_label: str
    is_active: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# 4. Detection Schemas
class DetectionCreate(BaseModel):
    media_asset_id: int
    class_name: str = Field(..., min_length=1, max_length=50)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBoxSchema
    frame_number: Optional[int] = Field(None, ge=0)
    image_reference: Optional[str] = None
    track_id: Optional[int] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    location_source: LocationSource = Field(
        default=LocationSource.SIMULATED,
        description="Source of coordinates e.g. simulated, gps_exif, manual_input, unknown"
    )
    review_status: ReviewStatusEnum = Field(default=ReviewStatusEnum.PENDING_REVIEW)

    @field_validator("class_name")
    @classmethod
    def validate_safety_class_name(cls, v: str) -> str:
        prohibited = ["confirmed survivor", "survivor confirmed", "alive", "deceased"]
        lower_v = v.strip().lower()
        for p in prohibited:
            if p in lower_v:
                raise ValueError(
                    f"Safety Violation: '{v}' violates Project Safety Rule 1. "
                    "Use 'person' or 'potential survivor'. Biological state cannot be confirmed by CV."
                )
        return lower_v


class DetectionRead(BaseModel):
    id: int
    media_asset_id: int
    track_id: Optional[int] = None
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    frame_number: Optional[int] = None
    image_reference: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_source: str
    review_status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# 5. PriorityAssessment Schemas
class PriorityAssessmentCreate(BaseModel):
    detection_id: int
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    priority_level: PriorityLevelEnum
    rationale: Optional[str] = None


class PriorityAssessmentRead(BaseModel):
    id: int
    detection_id: int
    urgency_score: float
    uncertainty_score: float
    priority_level: str
    rationale: Optional[str] = None
    assessed_at: datetime

    model_config = {"from_attributes": True}


# 6. Resource Schemas
class ResourceCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    resource_type: str = Field(..., min_length=2, max_length=50)
    total_capacity: int = Field(..., gt=0)
    available_capacity: int = Field(..., ge=0)
    simulated_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    simulated_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    is_available: bool = True

    @field_validator("available_capacity")
    @classmethod
    def validate_available_capacity(cls, v: int, info) -> int:
        if "total_capacity" in info.data and v > info.data["total_capacity"]:
            raise ValueError(f"available_capacity ({v}) cannot exceed total_capacity ({info.data['total_capacity']})")
        return v


class ResourceRead(BaseModel):
    id: int
    name: str
    resource_type: str
    total_capacity: int
    available_capacity: int
    simulated_lat: Optional[float] = None
    simulated_lon: Optional[float] = None
    is_available: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# 7. ResourceRecommendation Schemas
class ResourceRecommendationCreate(BaseModel):
    detection_id: int
    resource_id: int
    suitability_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=3)
    status: ReviewStatusEnum = Field(default=ReviewStatusEnum.PENDING_REVIEW)
    requires_human_authorization: bool = Field(
        default=True,
        description="Safety gate: strictly requires human review before dispatch"
    )


class ResourceRecommendationRead(BaseModel):
    id: int
    detection_id: int
    resource_id: int
    suitability_score: float
    rationale: str
    status: str
    requires_human_authorization: bool
    created_at: datetime

    model_config = {"from_attributes": True}
