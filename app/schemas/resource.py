"""
Pydantic Schemas for Emergency Resources and Advisory Recommendations.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import ResourceType, ReviewStatus


class ResourceBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    resource_type: ResourceType
    total_capacity: int = Field(..., gt=0, description="Total personnel count or unit slots")
    available_capacity: int = Field(..., ge=0)
    simulated_lat: float = Field(..., ge=-90.0, le=90.0)
    simulated_lon: float = Field(..., ge=-180.0, le=180.0)
    is_available: bool = True
    is_synthetic: bool = Field(True, description="Flag designating synthetic resource entity")


class ResourceCreate(ResourceBase):
    pass


class ResourceRead(ResourceBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AdvisoryRecommendationRead(BaseModel):
    id: int
    detection_id: int
    resource_id: int
    resource_name: str
    resource_type: ResourceType
    suitability_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    estimated_distance_m: float
    status: ReviewStatus = Field(ReviewStatus.PENDING_REVIEW)
    requires_human_authorization: bool = Field(
        True,
        description="Safety flag: Autonomous dispatch is forbidden; human confirmation mandatory."
    )
    created_at: datetime

    model_config = {"from_attributes": True}


class ResourceAllocationPlan(BaseModel):
    detection_id: int
    potential_survivor_count: int
    priority_level: str
    recommendations: List[AdvisoryRecommendationRead]
    advisory_notice: str = "ADVISORY ONLY. Strictly requires human coordinator verification before dispatch."
