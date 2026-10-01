"""
Pydantic Schemas for Priority and Verification Scoring (Phase 7).
Keeps Urgency (concern level) and Uncertainty (need for verification) strictly decoupled.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HazardSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"             # e.g. minor debris, puddle
    MODERATE = "MODERATE"   # e.g. blocked alleyway, shallow flood
    HIGH = "HIGH"           # e.g. partial collapse, rapid floodwaters
    CRITICAL = "CRITICAL"   # e.g. active fire, raging torrent, total structural collapse


class PriorityTier(str, Enum):
    CRITICAL_REVIEW = "CRITICAL_REVIEW"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class AssessmentInput(BaseModel):
    """Structured observations provided to the priority engine."""
    tracked_person_count: int = Field(default=0, ge=0, description="Count of distinct tracked persons")
    hazard_severity: HazardSeverity = Field(default=HazardSeverity.NONE, description="Visible hazard tier")
    is_accessible: Optional[bool] = Field(
        default=None,
        description="True if route is clear, False if blocked/flooded, None if unknown/unobserved"
    )
    average_detection_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Average model confidence for detected bounding boxes"
    )
    observation_age_minutes: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Minutes elapsed since the latest aerial frame observation"
    )
    has_missing_telemetry: bool = Field(
        default=False,
        description="True if UAV flight coordinates or altitude could not be resolved"
    )
    image_quality_degradation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Metric [0.0, 1.0] for smoke, dust, motion blur, or low lighting"
    )


class FactorExplanation(BaseModel):
    """Detailed breakdown explaining why a factor influenced the score."""
    factor_name: str
    raw_value: Any
    contribution: str
    weight_assumption: str
    impact_description: str


class PriorityAssessmentResult(BaseModel):
    """
    Transparent assessment output separating Urgency and Uncertainty.
    Every recommendation is advisory and strictly requires human review.
    """
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="Concern level [0.0, 1.0]")
    urgency_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    uncertainty_score: float = Field(..., ge=0.0, le=1.0, description="Need for verification [0.0, 1.0]")
    uncertainty_level: str = Field(..., description="LOW, MODERATE, or HIGH")
    composite_priority: PriorityTier = Field(..., description="Triaging tier for review queues")
    requires_human_verification: bool = Field(
        default=True,
        description="Safety rule: Human review is mandatory. Autonomous dispatch is forbidden."
    )
    contributing_factors: List[FactorExplanation]
    data_limitations: List[str] = Field(
        ...,
        description="Explicit documentation of missing, uncalibrated, or unmeasured signals"
    )
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def priority_category(self) -> PriorityTier:
        return self.composite_priority
