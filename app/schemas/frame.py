"""
Pydantic Schemas for Aerial Frames.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.telemetry import UAVTelemetryInput


class FrameBase(BaseModel):
    frame_index: int = Field(0, ge=0, description="Sequential frame index (for video) or 0 for static image")
    file_path: str = Field(..., description="Relative storage path of frame on disk")
    width_px: int = Field(..., gt=0)
    height_px: int = Field(..., gt=0)
    telemetry: Optional[UAVTelemetryInput] = None


class FrameCreate(FrameBase):
    mission_id: int


class FrameRead(FrameBase):
    id: int
    mission_id: int
    captured_at: datetime

    model_config = {"from_attributes": True}
