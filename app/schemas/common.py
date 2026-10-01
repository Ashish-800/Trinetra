"""
Common Enumerations and Primitive Schemas for the Disaster Assessment System.
"""
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class DisasterType(str, Enum):
    EARTHQUAKE = "EARTHQUAKE"
    FLOOD = "FLOOD"
    WILDFIRE = "WILDFIRE"
    HURRICANE = "HURRICANE"
    STRUCTURAL_COLLAPSE = "STRUCTURAL_COLLAPSE"
    GENERAL = "GENERAL"


class DetectionClass(str, Enum):
    PERSON = "person"
    POTENTIAL_SURVIVOR = "potential survivor"
    DEBRIS = "debris"
    FIRE = "fire"
    FLOOD_WATER = "flood_water"
    VEHICLE = "vehicle"
    COLLAPSED_STRUCTURE = "collapsed_structure"


class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    CRITICAL_REVIEW = "CRITICAL_REVIEW"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ResourceType(str, Enum):
    SAR_GROUND_TEAM = "SAR_GROUND_TEAM"
    MEDICAL_FIRST_RESPONDER = "MEDICAL_FIRST_RESPONDER"
    WATER_RESCUE_BOAT = "WATER_RESCUE_BOAT"
    HEAVY_EQUIPMENT = "HEAVY_EQUIPMENT"


class ReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    DISMISSED = "DISMISSED"


class BoundingBox(BaseModel):
    """Normalized or pixel bounding box coordinates."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @field_validator("x2")
    @classmethod
    def validate_x2(cls, v: float, info) -> float:
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError(f"x2 ({v}) must be greater than or equal to x1 ({info.data['x1']})")
        return v

    @field_validator("y2")
    @classmethod
    def validate_y2(cls, v: float, info) -> float:
        if "y1" in info.data and v < info.data["y1"]:
            raise ValueError(f"y2 ({v}) must be greater than or equal to y1 ({info.data['y1']})")
        return v


class Coordinates(BaseModel):
    """Geographic WGS-84 coordinates."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    is_synthetic: bool = Field(default=True, description="Flag explicitly indicating if coordinate is simulated/synthetic")
