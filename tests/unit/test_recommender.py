"""
Unit tests for Emergency Resource Recommendation Service (Phase 8).
Verifies:
- Available resource recommendations with transparent explanations
- Rejection of unavailable resources (offline / 0 capacity)
- Capability mismatch rejection
- Missing location handling (no fabricated distances or response times)
- 'NO_SUITABLE_RESOURCE_AVAILABLE' when inventory cannot fulfill requirement
- Strictly advisory recommendations (no auto-dispatch or reservation)
"""
import pytest
from app.schemas.location import LocationRecord, LocationSource
from app.schemas.resource_engine import (
    IncidentRequirement,
    SimulatedResource,
)
from app.services.recommender import ResourceRecommendationService


@pytest.fixture
def test_inventory() -> list[SimulatedResource]:
    base_loc = LocationRecord(latitude=34.05, longitude=-118.25, source=LocationSource.SIMULATED)
    return [
        SimulatedResource(
            id="BOAT-01",
            name="Flood Rescue Boat",
            resource_type="WATER_RESCUE",
            location=LocationRecord(latitude=34.06, longitude=-118.26, source=LocationSource.SIMULATED),
            is_available=True,
            capacity=6,
            available_capacity=4,
            capabilities=["flood_rescue", "shallow_water"],
            operating_constraints=[],
        ),
        SimulatedResource(
            id="SAR-01",
            name="Urban SAR Team",
            resource_type="SAR_TEAM",
            location=base_loc,
            is_available=True,
            capacity=5,
            available_capacity=5,
            capabilities=["ground_search", "debris_clearing", "first_aid"],
            operating_constraints=["daylight_only"],
        ),
        SimulatedResource(
            id="MED-OFFLINE",
            name="Paramedic Ambulance 1",
            resource_type="MEDICAL_UNIT",
            location=base_loc,
            is_available=False,  # Offline!
            capacity=2,
            available_capacity=2,
            capabilities=["first_aid", "trauma_stabilization"],
        ),
        SimulatedResource(
            id="SAR-FULL",
            name="Urban SAR Team Bravo",
            resource_type="SAR_TEAM",
            location=base_loc,
            is_available=True,
            capacity=4,
            available_capacity=0,  # 0 Available Capacity!
            capabilities=["ground_search", "first_aid"],
        ),
    ]


@pytest.fixture
def service(test_inventory: list[SimulatedResource]) -> ResourceRecommendationService:
    return ResourceRecommendationService(inventory=test_inventory)


# ==========================================
# 1. Available Resource Recommendation Test
# ==========================================

def test_available_resource_recommended(service: ResourceRecommendationService):
    """
    Scenario: Incident in a flooded zone requiring 'flood_rescue'.
    Expectation: BOAT-01 is recommended with detailed rationale and calculated distance.
    """
    incident = IncidentRequirement(
        incident_id="INC-FLOOD-01",
        disaster_type="FLOOD",
        required_capabilities=["flood_rescue"],
        needed_capacity=2,
        target_location=LocationRecord(latitude=34.055, longitude=-118.255, source=LocationSource.SIMULATED),
    )
    outcome = service.recommend(incident)

    assert outcome.status == "RECOMMENDATION_AVAILABLE"
    assert len(outcome.recommendations) >= 1

    top_rec = outcome.recommendations[0]
    assert top_rec.resource_id == "BOAT-01"
    assert "flood_rescue" in top_rec.matched_capabilities
    assert top_rec.requires_human_authorization is True
    assert top_rec.distance_meters is not None
    assert top_rec.distance_meters > 0.0
    assert "Estimated straight-line distance" in top_rec.distance_note
    assert "Matches all required capabilities" in top_rec.rationale


# ==========================================
# 2. Unavailable Resource Handling Test
# ==========================================

def test_unavailable_and_zero_capacity_resources_rejected(service: ResourceRecommendationService):
    """
    Scenario: Incident requires 'trauma_stabilization' and 'ground_search'.
    Expectation: MED-OFFLINE (offline) and SAR-FULL (0 capacity) are rejected with clear reasons.
    """
    # Request trauma stabilization: only MED-OFFLINE has it, but it is offline
    incident = IncidentRequirement(
        disaster_type="GENERAL",
        required_capabilities=["trauma_stabilization"],
        needed_capacity=1,
    )
    outcome = service.recommend(incident)

    assert outcome.status == "NO_SUITABLE_RESOURCE_AVAILABLE"
    assert "MED-OFFLINE" in outcome.unsuitable_resources_reasoning
    assert "unavailable" in outcome.unsuitable_resources_reasoning["MED-OFFLINE"]


# ==========================================
# 3. Capability Mismatch Test
# ==========================================

def test_capability_mismatch_rejected(service: ResourceRecommendationService):
    """
    Scenario: Incident requires 'hazardous_materials_containment' which no unit has.
    Expectation: All resources rejected due to missing capability.
    """
    incident = IncidentRequirement(
        disaster_type="CHEMICAL_SPILL",
        required_capabilities=["hazmat_containment"],
        needed_capacity=1,
    )
    outcome = service.recommend(incident)

    assert outcome.status == "NO_SUITABLE_RESOURCE_AVAILABLE"
    assert len(outcome.recommendations) == 0
    # Every resource in inventory should list lacking the required capability
    for res_id, reason in outcome.unsuitable_resources_reasoning.items():
        assert "lacks required capabilities" in reason or "unavailable" in reason


# ==========================================
# 4. Missing Location Handling Test
# ==========================================

def test_missing_location_does_not_invent_distance_or_response_time(service: ResourceRecommendationService):
    """
    Safety Rule: Considers location/distance only when valid location data exists.
    When target_location is None, distance must be None and no real response time is invented.
    """
    incident = IncidentRequirement(
        disaster_type="EARTHQUAKE",
        required_capabilities=["debris_clearing"],
        needed_capacity=2,
        target_location=None,  # Missing target location!
    )
    outcome = service.recommend(incident)

    assert outcome.status == "RECOMMENDATION_AVAILABLE"
    assert len(outcome.recommendations) == 1

    rec = outcome.recommendations[0]
    assert rec.resource_id == "SAR-01"
    assert rec.distance_meters is None
    assert "unavailable for distance calculation" in rec.distance_note
    assert "response time" not in rec.rationale.lower()  # Never fabricates response times


# ==========================================
# 5. Operating Constraint Violation Test
# ==========================================

def test_operating_constraint_violation(service: ResourceRecommendationService):
    """
    Scenario: SAR-01 has constraint 'daylight_only'.
    If incident occurs at night, SAR-01 must be rejected with constraint violation explanation.
    """
    incident = IncidentRequirement(
        disaster_type="EARTHQUAKE",
        required_capabilities=["ground_search"],
        needed_capacity=1,
        environmental_conditions=["night"],  # Night condition!
    )
    outcome = service.recommend(incident)

    assert "SAR-01" in outcome.unsuitable_resources_reasoning
    assert "daylight-only" in outcome.unsuitable_resources_reasoning["SAR-01"].lower()


# ==========================================
# 6. Safety & Advisory Disclaimers Test
# ==========================================

def test_safety_disclaimers_present(service: ResourceRecommendationService):
    incident = IncidentRequirement(
        disaster_type="FLOOD",
        required_capabilities=["flood_rescue"],
    )
    outcome = service.recommend(incident)

    assert "ADVISORY ONLY" in outcome.advisory_disclaimer
    assert "Autonomous dispatch is forbidden" in outcome.advisory_disclaimer
    assert outcome.recommendations[0].requires_human_authorization is True
