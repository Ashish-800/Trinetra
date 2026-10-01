"""
RescueNet Dataset Domain Schemas.
Maps 12-class (Background ID 0 + 11 foreground disaster classes IDs 1-11) aerial post-disaster
semantic segmentation to environmental hazards, terrain accessibility, and emergency resource requirements.
"""
from enum import IntEnum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.priority import HazardSeverity
from app.schemas.resource_engine import ResourceCapability


class RescueNetClass(IntEnum):
    """Semantic class definitions from RescueNet v1.0."""
    BACKGROUND = 0
    DEBRIS = 1
    WATER = 2
    BUILDING_NO_DAMAGE = 3
    BUILDING_MINOR_DAMAGE = 4
    BUILDING_MAJOR_DAMAGE = 5
    BUILDING_TOTAL_DESTRUCTION = 6
    VEHICLE = 7
    ROAD = 8
    TREE = 9
    POOL = 10
    SAND = 11


CLASS_DISPLAY_NAMES: Dict[RescueNetClass, str] = {
    RescueNetClass.BACKGROUND: "Background",
    RescueNetClass.DEBRIS: "Debris",
    RescueNetClass.WATER: "Water",
    RescueNetClass.BUILDING_NO_DAMAGE: "Building No Damage",
    RescueNetClass.BUILDING_MINOR_DAMAGE: "Building Minor Damage",
    RescueNetClass.BUILDING_MAJOR_DAMAGE: "Building Major Damage",
    RescueNetClass.BUILDING_TOTAL_DESTRUCTION: "Building Total Destruction",
    RescueNetClass.VEHICLE: "Vehicle",
    RescueNetClass.ROAD: "Road",
    RescueNetClass.TREE: "Tree",
    RescueNetClass.POOL: "Pool",
    RescueNetClass.SAND: "Sand",
}


class RescueNetHazardThresholds(BaseModel):
    """
    Configurable, transparent decision thresholds mapping RescueNet
    pixel distributions to standardized HazardSeverity tiers.
    RescueNet does NOT detect people/survivors; it strictly contextualizes environmental damage.
    """
    critical_destroyed_building_pct: float = 0.5
    critical_water_pct: float = 20.0
    high_major_building_pct: float = 1.0
    high_water_pct: float = 8.0
    high_debris_pct: float = 12.0
    moderate_debris_pct: float = 4.0
    moderate_water_pct: float = 2.0
    moderate_minor_building_pct: float = 3.0
    low_minor_building_pct: float = 0.5
    low_debris_pct: float = 1.0


class RescueNetSceneAnalysis(BaseModel):
    """
    Environmental scene analysis derived from a RescueNet aerial segmentation mask.
    Transforms pixel distributions into actionable decision-support parameters.
    NOTE: RescueNet provides macro-environmental disaster context (water, debris, collapse).
    It does NOT detect people or determine victim survival status.
    """
    image_id: str
    total_pixels: int
    class_distribution_pct: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage breakdown of each semantic class across the frame"
    )
    water_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    debris_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    destroyed_building_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    major_damage_building_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    minor_damage_building_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    intact_building_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    road_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    tree_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Vegetation/trees")
    pool_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Swimming pools")
    sand_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Beach/sand terrain")
    vehicle_detected: bool = False

    # Inferred Decision-Support Parameters
    inferred_hazard_severity: HazardSeverity = HazardSeverity.NONE
    inferred_accessibility: Optional[str] = Field(
        default="accessible",
        description="'accessible', 'difficult', 'isolated', or 'impassable'"
    )
    inferred_required_capabilities: List[str] = Field(
        default_factory=list,
        description="List of resource capabilities needed based on scene hazards"
    )
    contributing_hazard_factors: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations of detected aerial hazards"
    )
    thresholds_used: Optional[RescueNetHazardThresholds] = Field(
        default=None,
        description="Explicit documented thresholds used to derive hazard severity"
    )
