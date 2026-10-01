"""
Emergency Resource Recommendation Service.
Provides transparent, explainable decision support for matching emergency resources
to incident needs without autonomous dispatch or fabricated response times.
"""
import math
from typing import Any, List, Optional
from app.schemas.location import LocationRecord, LocationSource
from app.schemas.resource_engine import (
    IncidentRequirement,
    RecommendationOutcome,
    RecommendedResourceItem,
    SimulatedResource,
)


def compute_haversine_distance_m(loc1: LocationRecord, loc2: LocationRecord) -> float:
    """Computes straight-line great-circle distance between two geographic coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(loc1.latitude)
    phi2 = math.radians(loc2.latitude)
    delta_phi = math.radians(loc2.latitude - loc1.latitude)
    delta_lambda = math.radians(loc2.longitude - loc1.longitude)

    a = (
        math.sin(delta_phi / 2.0) ** 2 +
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 1)


def get_default_simulated_inventory() -> List[SimulatedResource]:
    """
    Returns a standard simulated emergency resource inventory for prototype demonstration.
    All resources are explicitly simulated.
    """
    base_lat, base_lon = 34.0522, -118.2437

    return [
        SimulatedResource(
            id="RES-BOAT-01",
            name="Zodiac Swiftwater Rescue Boat 1",
            resource_type="WATER_RESCUE",
            location=LocationRecord(
                latitude=base_lat + 0.010,
                longitude=base_lon - 0.008,
                source=LocationSource.SIMULATED,
                accuracy_meters=10.0,
            ),
            is_available=True,
            capacity=6,
            available_capacity=6,
            capabilities=["flood_rescue", "shallow_water", "water_evacuation", "night_operations"],
            operating_constraints=["requires_water_depth_0.5m"],
        ),
        SimulatedResource(
            id="RES-SAR-01",
            name="Urban SAR Ground Team Alpha",
            resource_type="SAR_TEAM",
            location=LocationRecord(
                latitude=base_lat - 0.005,
                longitude=base_lon + 0.005,
                source=LocationSource.SIMULATED,
                accuracy_meters=10.0,
            ),
            is_available=True,
            capacity=4,
            available_capacity=4,
            capabilities=["ground_search", "first_aid", "debris_clearing", "triage", "night_operations"],
            operating_constraints=["unpaved_road_inaccessible"],
        ),
        SimulatedResource(
            id="RES-MED-01",
            name="Mobile Paramedic Triage Unit 1",
            resource_type="MEDICAL_UNIT",
            location=LocationRecord(
                latitude=base_lat + 0.002,
                longitude=base_lon + 0.003,
                source=LocationSource.SIMULATED,
                accuracy_meters=5.0,
            ),
            is_available=True,
            capacity=2,
            available_capacity=2,
            capabilities=["advanced_life_support", "trauma_stabilization", "triage", "first_aid"],
            operating_constraints=["paved_roads_only"],
        ),
        SimulatedResource(
            id="RES-HEAVY-01",
            name="Technical Excavator & Crane Unit",
            resource_type="HEAVY_EQUIPMENT",
            location=LocationRecord(
                latitude=base_lat - 0.020,
                longitude=base_lon - 0.015,
                source=LocationSource.SIMULATED,
                accuracy_meters=20.0,
            ),
            is_available=False,  # Currently deployed / unavailable
            capacity=0,
            available_capacity=0,
            capabilities=["heavy_lifting", "structural_shoring", "rubble_clearing"],
            operating_constraints=["daylight_only", "requires_wide_clearance"],
        ),
    ]


class ResourceRecommendationService:
    """
    Evaluates incident requirements against available resources.
    Features:
    - Availability validation
    - Capability matching
    - Conditional distance computation (only when valid coordinates exist)
    - Explains rationales and returns 'NO_SUITABLE_RESOURCE_AVAILABLE' when appropriate
    - Strictly advisory: does not dispatch or invent response times
    """

    def __init__(self, inventory: Optional[List[SimulatedResource]] = None):
        self.inventory = inventory if inventory is not None else get_default_simulated_inventory()

    def recommend(
        self,
        incident: IncidentRequirement,
        inventory_override: Optional[List[SimulatedResource]] = None,
        max_recommendations: int = 3,
    ) -> RecommendationOutcome:
        active_inventory = inventory_override if inventory_override is not None else self.inventory
        suitable_candidates: List[tuple[SimulatedResource, float, List[str], Optional[float], str, str]] = []
        rejection_reasons: dict[str, str] = {}

        for resource in active_inventory:
            # 1. Availability check
            if not resource.is_available:
                rejection_reasons[resource.id] = f"Resource '{resource.name}' is currently marked unavailable (offline/deployed)."
                continue

            if resource.available_capacity <= 0:
                rejection_reasons[resource.id] = (
                    f"Resource '{resource.name}' is unavailable (0 available capacity slots)."
                )
                continue

            # 2. Capability matching check
            res_caps_lower = {c.lower() for c in resource.capabilities}
            req_caps_lower = {c.lower() for c in incident.required_capabilities}

            matched_caps = sorted(list(res_caps_lower.intersection(req_caps_lower)))
            missing_caps = req_caps_lower - res_caps_lower

            if missing_caps:
                rejection_reasons[resource.id] = (
                    f"Resource '{resource.name}' lacks required capabilities: {', '.join(sorted(missing_caps))}."
                )
                continue

            # 3. Operating constraints check
            constraint_violated = False
            for constraint in resource.operating_constraints:
                c_clean = constraint.lower().strip()
                # Check known environmental conflicts
                if "daylight_only" in c_clean and "night" in [e.lower() for e in incident.environmental_conditions]:
                    rejection_reasons[resource.id] = f"Constraint violated: '{resource.name}' operates daylight-only but incident is at night."
                    constraint_violated = True
                    break
                if "paved_roads_only" in c_clean and "impassable_roads" in [e.lower() for e in incident.environmental_conditions]:
                    rejection_reasons[resource.id] = f"Constraint violated: '{resource.name}' requires paved roads, but roads are reported impassable."
                    constraint_violated = True
                    break

            if constraint_violated:
                continue

            # 4. Location & Distance calculation (ONLY when both locations are valid)
            distance_m: Optional[float] = None
            distance_note: str

            if incident.target_location is not None and resource.location is not None:
                distance_m = compute_haversine_distance_m(incident.target_location, resource.location)
                dist_km = distance_m / 1000.0
                distance_note = f"Estimated straight-line distance: {dist_km:.2f} km ({distance_m:.0f} m)."
            else:
                distance_note = (
                    "Location data was unavailable for distance calculation; "
                    "resource was evaluated strictly on capability and capacity match."
                )

            # 5. Suitability scoring and rationale
            # Base capability score
            base_score = 0.70
            # Capacity bonus: full capacity fit
            if resource.available_capacity >= incident.needed_capacity:
                base_score += 0.15
            else:
                base_score += 0.05

            # Distance proximity factor (closer resources score higher, if distance is known)
            if distance_m is not None:
                # Up to 10 km: gradual penalty
                proximity_factor = max(0.0, 0.15 - (distance_m / 100000.0))
                base_score += proximity_factor

            suitability_score = round(min(1.0, max(0.1, base_score)), 2)

            rationale_parts = [
                f"Matches all required capabilities ({', '.join(incident.required_capabilities)}).",
                f"Available capacity ({resource.available_capacity} slots) for needed capacity ({incident.needed_capacity}).",
                distance_note,
            ]
            full_rationale = " ".join(rationale_parts)

            suitable_candidates.append(
                (resource, suitability_score, matched_caps, distance_m, distance_note, full_rationale)
            )

        # Sort candidates: primarily by suitability score descending, then by distance ascending if available
        def sort_key(item):
            res_obj, score, _, dist, _, _ = item
            dist_val = dist if dist is not None else 9999999.0
            return (-score, dist_val)

        suitable_candidates.sort(key=sort_key)

        # Build output items
        if not suitable_candidates:
            summary = (
                f"No suitable resource available for disaster '{incident.disaster_type}' "
                f"requiring capabilities: {', '.join(incident.required_capabilities) if incident.required_capabilities else 'None specified'}."
            )
            return RecommendationOutcome(
                status="NO_SUITABLE_RESOURCE_AVAILABLE",
                incident_summary=summary,
                recommendations=[],
                unsuitable_resources_reasoning=rejection_reasons,
            )

        recommended_items: List[RecommendedResourceItem] = []
        for resource, score, matched_caps, dist_m, dist_note, rationale in suitable_candidates[:max_recommendations]:
            recommended_items.append(
                RecommendedResourceItem(
                    resource_id=resource.id,
                    resource_name=resource.name,
                    resource_type=resource.resource_type,
                    suitability_score=score,
                    matched_capabilities=matched_caps,
                    distance_meters=dist_m,
                    distance_note=dist_note,
                    rationale=rationale,
                    requires_human_authorization=True,
                )
            )

        summary = (
            f"Identified {len(recommended_items)} candidate resource(s) matching requirements "
            f"for '{incident.disaster_type}' incident."
        )

        return RecommendationOutcome(
            status="RECOMMENDATION_AVAILABLE",
            incident_summary=summary,
            recommendations=recommended_items,
            unsuitable_resources_reasoning=rejection_reasons,
        )

    def recommend_resources(
        self,
        requirement: IncidentRequirement,
        inventory: Optional[List[Any]] = None,
        **kwargs,
    ) -> RecommendationOutcome:
        """Alias for recommend() supporting benchmark and workflow evaluation callers."""
        converted_inventory = None
        if inventory is not None:
            converted_inventory = []
            for item in inventory:
                if isinstance(item, SimulatedResource):
                    converted_inventory.append(item)
                else:
                    loc = None
                    if getattr(item, "simulated_lat", None) is not None and getattr(item, "simulated_lon", None) is not None:
                        loc = LocationRecord(
                            latitude=item.simulated_lat,
                            longitude=item.simulated_lon,
                            source=LocationSource.SIMULATED,
                        )
                    caps = [
                        c.value if hasattr(c, "value") else str(c)
                        for c in getattr(item, "capabilities", [])
                    ]
                    converted_inventory.append(
                        SimulatedResource(
                            id=str(getattr(item, "resource_id", getattr(item, "id", "RES"))),
                            name=getattr(item, "name", "Resource"),
                            resource_type=str(getattr(item, "resource_type", "GENERAL")),
                            location=loc,
                            is_available=getattr(item, "is_available", True),
                            capacity=getattr(item, "total_capacity", getattr(item, "capacity", 1)),
                            available_capacity=getattr(item, "available_capacity", 1),
                            capabilities=caps,
                            operating_constraints=getattr(item, "operating_constraints", []),
                        )
                    )
        return self.recommend(
            incident=requirement,
            inventory_override=converted_inventory,
            **kwargs,
        )
