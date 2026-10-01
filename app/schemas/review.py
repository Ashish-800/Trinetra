"""
Pydantic Schemas for Human Review and Verification Actions.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.common import PriorityLevel, ReviewStatus


class HumanReviewAction(BaseModel):
    """Payload submitted by a human operator verifying or updating a detection."""
    status: ReviewStatus = Field(..., description="CONFIRMED, REJECTED, or DISMISSED")
    reviewer_name: str = Field(..., min_length=2, max_length=80, description="Operator callsign or identifier")
    notes: Optional[str] = Field(None, max_length=500, description="Operator assessment notes")
    override_priority: Optional[PriorityLevel] = Field(
        None, description="Optional operator reassessment of priority tier"
    )


class HumanReviewRecord(BaseModel):
    id: int
    detection_id: int
    reviewer_name: str
    action: ReviewStatus
    notes: Optional[str]
    reviewed_at: datetime

    model_config = {"from_attributes": True}
