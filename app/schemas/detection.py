"""
Pydantic Schemas for Detections and Spatial Queries.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.common import BoundingBox, PriorityLevel, ReviewStatus


class DetectionBase(BaseModel):
    class_name: str = Field(..., description="Detected category e.g. 'person', 'debris', 'flood_water'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model detection confidence score")
    bbox: BoundingBox
    tracking_id: Optional[int] = Field(None, description="Consistent tracking ID assigned across sequential frames")
    
    # Geolocation estimation
    estimated_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    estimated_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    uncertainty_radius_m: Optional[float] = Field(None, ge=0.0, description="Spatial margin of error in meters")
    is_synthetic: bool = Field(True, description="Flag explicitly indicating simulated or synthetic data")

    # Scoring & Prioritization
    urgency_score: float = Field(0.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(0.0, ge=0.0, le=1.0)
    priority_level: PriorityLevel = Field(PriorityLevel.LOW)
    review_status: ReviewStatus = Field(ReviewStatus.PENDING_REVIEW)

    @field_validator("class_name")
    @classmethod
    def validate_safety_class_name(cls, v: str) -> str:
        prohibited = ["confirmed survivor", "survivor confirmed", "alive", "deceased"]
        lower_v = v.strip().lower()
        for p in prohibited:
            if p in lower_v:
                raise ValueError(
                    f"Safety Violation: '{v}' is prohibited by the Project Constitution. "
                    "Use 'person' or 'potential survivor'. Biological state cannot be confirmed by CV."
                )
        return lower_v


class DetectionCreate(DetectionBase):
    frame_id: int


class DetectionRead(DetectionBase):
    id: int
    frame_id: int
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None

    model_config = {"from_attributes": True}


class DetectionFilter(BaseModel):
    class_name: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    priority_level: Optional[PriorityLevel] = None
    review_status: Optional[ReviewStatus] = None
    min_urgency: Optional[float] = Field(None, ge=0.0, le=1.0)
