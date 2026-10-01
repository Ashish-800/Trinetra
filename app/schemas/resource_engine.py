"""
Pydantic Schemas for Resource Recommendation Service (Phase 8).
Supports simulated resource inventory, capability matching,
conditional distance evaluation, and transparent advisory recommendations.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from app.schemas.location import LocationRecord, LocationSource


class ResourceCapability(str, Enum):
    """Standard emergency response and disaster relief capability classifications."""
    WATER_RESCUE = "WATER_RESCUE"
    HEAVY_DEBRIS_CLEARING = "HEAVY_DEBRIS_CLEARING"
    CANINE_SEARCH = "CANINE_SEARCH"
    STRUCTURAL_SHORING = "STRUCTURAL_SHORING"
    USAR_TEAM = "USAR_TEAM"
    GROUND_TRANSPORT = "GROUND_TRANSPORT"
    TRIAGE_MEDICAL = "TRIAGE_MEDICAL"
    FLOOD_RESCUE = "flood_rescue"
    SHALLOW_WATER = "shallow_water"
    FIRST_AID = "first_aid"
    DEBRIS_CLEARING = "debris_clearing"
    GROUND_SEARCH = "ground_search"
    TRAUMA_STABILIZATION = "trauma_stabilization"


class SimulatedResourceInventoryItem(BaseModel):
    """A resource inventory record for benchmark scenarios and evaluation suites."""
    resource_id: Union[int, str]
    name: str
    resource_type: str
    capabilities: List[Union[ResourceCapability, str]] = Field(default_factory=list)
    total_capacity: int = Field(default=1, gt=0)
    available_capacity: int = Field(default=1, ge=0)
    is_available: bool = True
    simulated_lat: Optional[float] = None
    simulated_lon: Optional[float] = None
    operating_constraints: List[str] = Field(default_factory=list)


class SimulatedResource(BaseModel):
    """Simulated emergency rescue resource in inventory."""
    id: str = Field(..., description="Unique resource ID, e.g. 'RES-BOAT-01'")
    name: str = Field(..., min_length=2, max_length=120)
    resource_type: str = Field(..., description="Category, e.g. WATER_RESCUE, SAR_TEAM, MEDICAL_UNIT")
    location: Optional[LocationRecord] = Field(default=None, description="Current simulated resource location")
    is_available: bool = Field(default=True, description="True if unit is not deployed/offline")
    capacity: int = Field(..., ge=0, description="Total person rescue/transport capacity")
    available_capacity: int = Field(..., ge=0, description="Remaining available capacity slots")
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capability tags, e.g. ['flood_rescue', 'shallow_water', 'first_aid']"
    )
    operating_constraints: List[str] = Field(
        default_factory=list,
        description="Operational limits, e.g. ['daylight_only', 'max_wind_knots_25']"
    )

    @field_validator("available_capacity")
    @classmethod
    def validate_available_capacity(cls, v: int, info) -> int:
        if "capacity" in info.data and v > info.data["capacity"]:
            raise ValueError(f"available_capacity ({v}) cannot exceed total capacity ({info.data['capacity']})")
        return v


class IncidentRequirement(BaseModel):
    """Incident characteristics and required support parameters."""
    incident_id: Optional[str] = Field(default=None, description="Identifier of the incident or zone")
    disaster_type: str = Field(default="GENERAL", description="Disaster type, e.g. FLOOD, COLLAPSE, WILDFIRE")
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="Capabilities strictly needed, e.g. ['flood_rescue']"
    )
    needed_capacity: int = Field(default=1, gt=0, description="Number of persons needing assistance")
    target_location: Optional[LocationRecord] = Field(
        default=None,
        description="Estimated location of incident (may be None if uncalibrated)"
    )
    urgency_level: Optional[str] = Field(default="MEDIUM", description="CRITICAL, HIGH, MEDIUM, LOW")
    environmental_conditions: List[str] = Field(
        default_factory=list,
        description="Current environmental flags, e.g. ['night', 'rough_water', 'impassable_roads']"
    )

    # Compatibility fields for benchmark scenarios and RescueNet caller conventions
    required_capability: Optional[Union[ResourceCapability, str]] = None
    required_capacity: Optional[int] = None
    incident_lat: Optional[float] = None
    incident_lon: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.required_capability is not None:
            cap_val = (
                self.required_capability.value
                if isinstance(self.required_capability, Enum)
                else str(self.required_capability)
            )
            if cap_val not in self.required_capabilities:
                self.required_capabilities.append(cap_val)
        if self.required_capacity is not None and self.needed_capacity == 1:
            self.needed_capacity = self.required_capacity
        if self.target_location is None and self.incident_lat is not None and self.incident_lon is not None:
            self.target_location = LocationRecord(
                latitude=self.incident_lat,
                longitude=self.incident_lon,
                source=LocationSource.SIMULATED,
            )


class RecommendedResourceItem(BaseModel):
    """An individual advisory resource recommendation."""
    resource_id: str
    resource_name: str
    resource_type: str
    suitability_score: float = Field(..., ge=0.0, le=1.0)
    matched_capabilities: List[str]
    distance_meters: Optional[float] = Field(
        default=None,
        description="Calculated straight-line distance (None if location is missing)"
    )
    distance_note: str = Field(
        ...,
        description="Explanation of distance calculation or statement that location was unavailable"
    )
    rationale: str = Field(..., description="Explainable rationale for recommendation")
    status: str = Field(
        default="PENDING_REVIEW",
        description="Advisory workflow status: strictly PENDING_REVIEW pending human commander approval"
    )
    requires_human_authorization: bool = Field(
        default=True,
        description="Safety boundary: Autonomous dispatch is forbidden. Operator confirmation required."
    )


class RecommendationOutcome(BaseModel):
    """Comprehensive recommendation output."""
    status: str = Field(
        ...,
        description="'RECOMMENDATION_AVAILABLE' or 'NO_SUITABLE_RESOURCE_AVAILABLE'"
    )
    incident_summary: str
    recommendations: List[RecommendedResourceItem] = Field(default_factory=list)
    unsuitable_resources_reasoning: Dict[str, str] = Field(
        default_factory=dict,
        description="Explanations for why specific resources in inventory were not selected"
    )
    advisory_disclaimer: str = Field(
        default=(
            "ADVISORY ONLY: Autonomous dispatch is forbidden. Recommendations do not constitute an automatic dispatch or reservation. "
            "Real response times are not fabricated; deployment requires human coordinator authorization."
        )
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
