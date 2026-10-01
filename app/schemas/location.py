"""
Location Schemas and Validation Contracts.
Enforces strict source tagging (simulated vs real), coordinate bounds,
and uncertainty/accuracy metadata.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class LocationSource(str, Enum):
    SIMULATED = "simulated"
    REAL = "real"
    UNKNOWN = "unknown"


class LocationRecord(BaseModel):
    """
    Standard location representation for all spatial entities.
    Must include latitude, longitude, and explicit source designation.
    """
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS-84 Latitude [-90.0, 90.0]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS-84 Longitude [-180.0, 180.0]")
    source: LocationSource = Field(
        default=LocationSource.SIMULATED,
        description="Explicit source: 'simulated' or 'real'"
    )
    accuracy_meters: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated spatial error radius in meters (uncertainty bound)"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when coordinate was captured or computed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata (e.g. altitude, projection method, sensor tags)"
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: LocationSource) -> LocationSource:
        if v not in (LocationSource.SIMULATED, LocationSource.REAL, LocationSource.UNKNOWN):
            raise ValueError(f"Invalid location source: {v}. Must be 'simulated', 'real', or 'unknown'.")
        return v


class DroneTelemetryMetadata(BaseModel):
    """
    Interface for UAV sensor telemetry.
    Does not connect to hardware; serves as a typed contract for spatial estimation.
    """
    drone_latitude: float = Field(..., ge=-90.0, le=90.0, description="UAV GPS Latitude")
    drone_longitude: float = Field(..., ge=-180.0, le=180.0, description="UAV GPS Longitude")
    altitude_agl_m: float = Field(..., gt=0.0, description="Altitude Above Ground Level in meters")
    gimbal_pitch_deg: float = Field(
        -90.0,
        ge=-90.0,
        le=30.0,
        description="Gimbal pitch angle in degrees (-90 is nadir / straight down)"
    )
    gimbal_yaw_deg: float = Field(
        0.0,
        ge=0.0,
        lt=360.0,
        description="Gimbal yaw heading relative to True North [0, 360)"
    )
    horizontal_fov_deg: float = Field(
        84.0,
        gt=10.0,
        lt=180.0,
        description="Camera horizontal field of view in degrees"
    )
    source: LocationSource = Field(
        default=LocationSource.SIMULATED,
        description="Telemetry source (default 'simulated' for prototype)"
    )
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
