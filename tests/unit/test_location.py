"""
Unit tests for Location Handling Service (Phase 6).
Verifies:
- Valid coordinate validation
- Invalid coordinate rejection
- Missing location handling (no coordinates inferred from pixels alone)
- Simulated-source labeling
- Telemetry offset vs. drone coordinate (drone pos is not assumed to be person pos)
- Accuracy error radii and timestamps
"""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from app.schemas.domain import BoundingBoxSchema
from app.schemas.location import DroneTelemetryMetadata, LocationRecord, LocationSource
from app.services.location import (
    InvalidCoordinateError,
    PlanarRayCastingEstimator,
    SimulatedLocationProvider,
)


# ==========================================
# 1. Coordinate Validation Tests
# ==========================================

def test_valid_location_record():
    loc = LocationRecord(
        latitude=34.0522,
        longitude=-118.2437,
        source=LocationSource.SIMULATED,
        accuracy_meters=12.5,
        metadata={"mission_id": 1},
    )
    assert loc.latitude == 34.0522
    assert loc.longitude == -118.2437
    assert loc.source == LocationSource.SIMULATED
    assert loc.accuracy_meters == 12.5
    assert loc.timestamp is not None


def test_invalid_latitude_out_of_bounds():
    with pytest.raises(ValidationError):
        LocationRecord(
            latitude=95.0,  # Invalid: > 90.0
            longitude=0.0,
            source=LocationSource.SIMULATED,
        )

    with pytest.raises(ValidationError):
        LocationRecord(
            latitude=-90.01,  # Invalid: < -90.0
            longitude=0.0,
            source=LocationSource.SIMULATED,
        )


def test_invalid_longitude_out_of_bounds():
    with pytest.raises(ValidationError):
        LocationRecord(
            latitude=0.0,
            longitude=180.5,  # Invalid: > 180.0
            source=LocationSource.SIMULATED,
        )

    with pytest.raises(ValidationError):
        LocationRecord(
            latitude=0.0,
            longitude=-181.0,  # Invalid: < -180.0
            source=LocationSource.SIMULATED,
        )


def test_invalid_accuracy_negative():
    with pytest.raises(ValidationError):
        LocationRecord(
            latitude=0.0,
            longitude=0.0,
            accuracy_meters=-5.0,  # Invalid: < 0
        )


def test_simulated_location_provider_invalid_base_coords():
    with pytest.raises(InvalidCoordinateError):
        SimulatedLocationProvider(base_latitude=99.0, base_longitude=0.0)

    with pytest.raises(InvalidCoordinateError):
        SimulatedLocationProvider(base_latitude=0.0, base_longitude=200.0)


# ==========================================
# 2. Simulated Source Labeling Tests
# ==========================================

def test_simulated_location_provider_strictly_tags_simulated():
    """Safety Rule: Prototype locations must be explicitly tagged as simulated."""
    provider = SimulatedLocationProvider(base_latitude=28.6139, base_longitude=77.2090)
    bbox = BoundingBoxSchema(x1=200, y1=150, x2=240, y2=210)

    loc = provider.estimate_location(bbox, image_width=800, image_height=600)

    assert loc is not None
    assert loc.source == LocationSource.SIMULATED
    assert loc.source == "simulated"
    assert loc.accuracy_meters is not None
    assert loc.accuracy_meters > 0.0
    assert loc.metadata["is_synthetic"] is True
    assert "simulated" in loc.metadata["note"].lower()


# ==========================================
# 3. Missing Telemetry Tests (No GPS from Pixels Alone)
# ==========================================

def test_missing_telemetry_returns_none():
    """
    Safety Rule: Do NOT infer a person's real-world GPS position from image pixels alone.
    If telemetry is None, PlanarRayCastingEstimator must return None.
    """
    estimator = PlanarRayCastingEstimator()
    bbox = BoundingBoxSchema(x1=100, y1=100, x2=150, y2=180)

    result = estimator.estimate_location(
        bbox=bbox,
        image_width=1920,
        image_height=1080,
        telemetry=None,  # Missing telemetry!
    )

    assert result is None, "Estimator must not invent GPS coordinates without flight telemetry."


# ==========================================
# 4. Drone Position vs. Person Ground Position
# ==========================================

def test_drone_coordinates_not_simply_copied_as_person_coordinates():
    """
    Safety Rule: Do not pretend that the drone's GPS coordinate is automatically
    the person's exact coordinate. Ground projection must compute spatial offsets.
    """
    estimator = PlanarRayCastingEstimator()
    # Person is off-center in the top-right quadrant of the image
    bbox = BoundingBoxSchema(x1=1400, y1=200, x2=1450, y2=300)

    telemetry = DroneTelemetryMetadata(
        drone_latitude=34.052200,
        drone_longitude=-118.243700,
        altitude_agl_m=60.0,
        gimbal_pitch_deg=-45.0,  # Oblique 45-degree angle
        gimbal_yaw_deg=45.0,     # Heading North-East
        horizontal_fov_deg=84.0,
        source=LocationSource.SIMULATED,
        timestamp=datetime.now(timezone.utc),
    )

    loc = estimator.estimate_location(
        bbox=bbox,
        image_width=1920,
        image_height=1080,
        telemetry=telemetry,
    )

    assert loc is not None
    assert loc.source == LocationSource.SIMULATED

    # The estimated person location must differ from the drone's location due to ray projection
    assert loc.latitude != telemetry.drone_latitude
    assert loc.longitude != telemetry.drone_longitude

    # Accuracy / uncertainty bound must be reported
    assert loc.accuracy_meters is not None
    assert loc.accuracy_meters >= 10.0  # At 60m altitude and 45 deg, error radius must reflect uncertainty

    # Contextual metadata must record the ground offset and methodology
    assert "ground_offset_meters" in loc.metadata
    assert loc.metadata["ground_offset_meters"] > 0.0
    assert loc.metadata["drone_latitude"] == telemetry.drone_latitude
    assert loc.metadata["drone_longitude"] == telemetry.drone_longitude
    assert loc.timestamp == telemetry.timestamp
