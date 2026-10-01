"""
Pydantic Schemas for Missions.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.common import DisasterType


class MissionBase(BaseModel):
    title: str = Field(..., min_length=3, max_length=120, description="Mission identifier or title")
    disaster_type: DisasterType = Field(default=DisasterType.GENERAL, description="Disaster category")
    location_name: str = Field(..., max_length=200, description="Area of operations or sector name")
    simulated_base_lat: float = Field(..., ge=-90.0, le=90.0, description="Base camp / takeoff latitude")
    simulated_base_lon: float = Field(..., ge=-180.0, le=180.0, description="Base camp / takeoff longitude")
    notes: Optional[str] = Field(None, max_length=1000)


class MissionCreate(MissionBase):
    pass


class MissionUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=120)
    disaster_type: Optional[DisasterType] = None
    location_name: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class MissionRead(MissionBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
