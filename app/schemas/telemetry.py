"""
Pydantic Schemas for UAV Telemetry and Flight Sensor Readings.
"""
from pydantic import BaseModel, Field


class UAVTelemetryInput(BaseModel):
    """Telemetry captured or simulated during image/video frame acquisition."""
    uav_lat: float = Field(..., ge=-90.0, le=90.0, description="UAV GPS Latitude")
    uav_lon: float = Field(..., ge=-180.0, le=180.0, description="UAV GPS Longitude")
    altitude_m: float = Field(..., gt=0.0, le=1000.0, description="Altitude Above Ground Level (AGL) in meters")
    gimbal_pitch_deg: float = Field(-90.0, ge=-90.0, le=30.0, description="Camera pitch angle (-90 is nadir / straight down)")
    gimbal_yaw_deg: float = Field(0.0, ge=-180.0, le=360.0, description="Camera yaw / heading relative to North")
    focal_length_mm: float = Field(24.0, gt=0.0, description="Camera lens equivalent focal length in mm")
    fov_deg: float = Field(84.0, gt=10.0, lt=180.0, description="Horizontal field of view in degrees")
    is_synthetic: bool = Field(True, description="Explicit flag designating whether telemetry is simulated or test data")
