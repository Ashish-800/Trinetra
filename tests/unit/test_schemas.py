"""
Unit tests for Pydantic schemas and Safety Constitution validation rules.
"""
import pytest
from pydantic import ValidationError
from app.schemas.common import (
    BoundingBox,
    Coordinates,
    DisasterType,
    PriorityLevel,
    ResourceType,
    ReviewStatus,
)
from app.schemas.detection import DetectionBase
from app.schemas.mission import MissionCreate
from app.schemas.resource import ResourceCreate
from app.schemas.review import HumanReviewAction
from app.schemas.telemetry import UAVTelemetryInput


def test_bounding_box_valid():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
    assert bbox.x1 == 10.0
    assert bbox.y1 == 20.0
    assert bbox.x2 == 50.0
    assert bbox.y2 == 80.0


def test_bounding_box_invalid_x2_less_than_x1():
    with pytest.raises(ValidationError) as exc_info:
        BoundingBox(x1=100.0, y1=20.0, x2=50.0, y2=80.0)
    assert "x2" in str(exc_info.value)


def test_bounding_box_invalid_y2_less_than_y1():
    with pytest.raises(ValidationError) as exc_info:
        BoundingBox(x1=10.0, y1=100.0, x2=50.0, y2=80.0)
    assert "y2" in str(exc_info.value)


def test_coordinates_valid_and_synthetic_flag():
    coords = Coordinates(latitude=34.05, longitude=-118.25)
    assert coords.latitude == 34.05
    assert coords.longitude == -118.25
    assert coords.is_synthetic is True  # Default must be True for safety


def test_coordinates_out_of_bounds():
    with pytest.raises(ValidationError):
        Coordinates(latitude=95.0, longitude=0.0)
    with pytest.raises(ValidationError):
        Coordinates(latitude=0.0, longitude=-190.0)


def test_safety_constitution_rejects_confirmed_survivor():
    """Safety Rule 1: Never permit 'confirmed survivor' in detection class_name."""
    prohibited_names = [
        "confirmed survivor",
        "Confirmed Survivor",
        "survivor confirmed",
        "alive person",
        "deceased person",
    ]
    for prohibited in prohibited_names:
        with pytest.raises(ValidationError) as exc_info:
            DetectionBase(
                class_name=prohibited,
                confidence=0.95,
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )
        assert "Safety Violation" in str(exc_info.value)


def test_detection_valid_person_label():
    """Permits 'person' and 'potential survivor'."""
    det1 = DetectionBase(
        class_name="person",
        confidence=0.88,
        bbox=BoundingBox(x1=10, y1=15, x2=45, y2=80),
    )
    assert det1.class_name == "person"
    assert det1.is_synthetic is True

    det2 = DetectionBase(
        class_name="potential survivor",
        confidence=0.75,
        bbox=BoundingBox(x1=5, y1=5, x2=20, y2=30),
    )
    assert det2.class_name == "potential survivor"


def test_telemetry_schema_validation():
    telem = UAVTelemetryInput(
        uav_lat=35.6895,
        uav_lon=139.6917,
        altitude_m=50.0,
        gimbal_pitch_deg=-45.0,
    )
    assert telem.altitude_m == 50.0
    assert telem.is_synthetic is True

    with pytest.raises(ValidationError):
        # Negative altitude not allowed
        UAVTelemetryInput(
            uav_lat=35.6895,
            uav_lon=139.6917,
            altitude_m=-5.0,
        )


def test_mission_create_schema():
    mission = MissionCreate(
        title="Sector A Flood Recon",
        disaster_type=DisasterType.FLOOD,
        location_name="River Valley",
        simulated_base_lat=29.9792,
        simulated_base_lon=31.1342,
    )
    assert mission.title == "Sector A Flood Recon"
    assert mission.disaster_type == DisasterType.FLOOD


def test_resource_create_schema():
    res = ResourceCreate(
        name="Rescue Boat Bravo",
        resource_type=ResourceType.WATER_RESCUE_BOAT,
        total_capacity=6,
        available_capacity=6,
        simulated_lat=29.98,
        simulated_lon=31.14,
    )
    assert res.resource_type == ResourceType.WATER_RESCUE_BOAT
    assert res.is_synthetic is True


def test_human_review_action_schema():
    action = HumanReviewAction(
        status=ReviewStatus.CONFIRMED,
        reviewer_name="Operator Johnson",
        notes="Person spotted on roof, verified against high-res crop",
        override_priority=PriorityLevel.CRITICAL_REVIEW,
    )
    assert action.status == ReviewStatus.CONFIRMED
    assert action.reviewer_name == "Operator Johnson"
    assert action.override_priority == PriorityLevel.CRITICAL_REVIEW
