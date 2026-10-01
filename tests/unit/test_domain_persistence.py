"""
Unit tests for Domain Schemas and SQLAlchemy Persistence (Phase 2).
Entities tested:
- Incident
- MediaAsset
- Detection
- Track
- PriorityAssessment
- Resource
- ResourceRecommendation
"""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session
from app.models.domain import (
    Detection,
    Incident,
    MediaAsset,
    PriorityAssessment,
    Resource,
    ResourceRecommendation,
    Track,
)
from app.schemas.domain import (
    BoundingBoxSchema,
    DetectionCreate,
    IncidentCreate,
    LocationSource,
    MediaAssetCreate,
    PriorityAssessmentCreate,
    PriorityLevelEnum,
    ResourceCreate,
    ResourceRecommendationCreate,
    ReviewStatusEnum,
    TrackCreate,
)


# ==========================================
# 1. Validation Tests (Valid vs Invalid Fields)
# ==========================================

def test_incident_schema_validation():
    # Valid
    inc = IncidentCreate(title="Operation River Flood", disaster_type="FLOOD", location_name="Sector 4")
    assert inc.title == "Operation River Flood"

    # Invalid title (too short)
    with pytest.raises(ValidationError):
        IncidentCreate(title="AB", location_name="Sector 4")


def test_media_asset_schema_validation():
    # Valid image
    asset = MediaAssetCreate(
        incident_id=1,
        file_path="uploads/test_asset.jpg",
        media_type="IMAGE",
        mime_type="image/jpeg",
        file_size_bytes=1024,
        width_px=1920,
        height_px=1080,
    )
    assert asset.media_type == "IMAGE"

    # Invalid media_type
    with pytest.raises(ValidationError):
        MediaAssetCreate(
            incident_id=1,
            file_path="test.mp3",
            media_type="AUDIO",  # unsupported
            mime_type="audio/mp3",
            file_size_bytes=100,
        )


def test_detection_schema_validation_and_safety_rules():
    bbox = BoundingBoxSchema(x1=10.0, y1=15.0, x2=50.0, y2=80.0)

    # Valid detection with "simulated" location source
    det = DetectionCreate(
        media_asset_id=1,
        class_name="person",
        confidence=0.88,
        bbox=bbox,
        frame_number=12,
        image_reference="uploads/frames/frame_0012.jpg",
        latitude=34.0522,
        longitude=-118.2437,
        location_source=LocationSource.SIMULATED,
        review_status=ReviewStatusEnum.PENDING_REVIEW,
    )
    assert det.location_source == "simulated"
    assert det.class_name == "person"

    # Safety Rule 1: Reject "confirmed survivor"
    with pytest.raises(ValidationError) as exc_info:
        DetectionCreate(
            media_asset_id=1,
            class_name="confirmed survivor",
            confidence=0.99,
            bbox=bbox,
        )
    assert "Safety Violation" in str(exc_info.value)

    # Invalid confidence (> 1.0)
    with pytest.raises(ValidationError):
        DetectionCreate(
            media_asset_id=1,
            class_name="person",
            confidence=1.5,
            bbox=bbox,
        )

    # Invalid coordinates (latitude > 90)
    with pytest.raises(ValidationError):
        DetectionCreate(
            media_asset_id=1,
            class_name="person",
            confidence=0.8,
            bbox=bbox,
            latitude=105.0,
            longitude=0.0,
        )


def test_resource_schema_validation():
    # Valid
    res = ResourceCreate(
        name="Urban SAR Team Alpha",
        resource_type="SAR_GROUND_TEAM",
        total_capacity=10,
        available_capacity=8,
        simulated_lat=34.0,
        simulated_lon=-118.0,
    )
    assert res.available_capacity == 8

    # Invalid: available capacity exceeds total capacity
    with pytest.raises(ValidationError) as exc_info:
        ResourceCreate(
            name="Urban SAR Team Alpha",
            resource_type="SAR_GROUND_TEAM",
            total_capacity=10,
            available_capacity=15,
        )
    assert "cannot exceed total_capacity" in str(exc_info.value)


# ==========================================
# 2. Database Persistence & Relationship Tests
# ==========================================

def test_save_and_retrieve_incident_and_media_asset(db_session: Session):
    incident = Incident(
        title="Hurricane Surge Delta",
        disaster_type="HURRICANE",
        location_name="Coastal Marina",
        notes="High wind and tidal surge",
    )
    db_session.add(incident)
    db_session.commit()
    db_session.refresh(incident)

    assert incident.id is not None
    assert incident.created_at is not None

    media = MediaAsset(
        incident_id=incident.id,
        file_path="uploads/surveys/coastal_sweep.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        file_size_bytes=5242880,
        width_px=1920,
        height_px=1080,
        duration_seconds=30.0,
        frame_count=900,
    )
    db_session.add(media)
    db_session.commit()
    db_session.refresh(media)

    assert media.id is not None
    assert len(incident.media_assets) == 1
    assert incident.media_assets[0].file_path == "uploads/surveys/coastal_sweep.mp4"


def test_track_and_detection_relationships(db_session: Session):
    incident = Incident(title="Earthquake Recon", location_name="Sector 7")
    db_session.add(incident)
    db_session.commit()

    media = MediaAsset(
        incident_id=incident.id,
        file_path="uploads/flight_01.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        file_size_bytes=10000,
        width_px=1280,
        height_px=720,
    )
    track = Track(
        incident_id=incident.id,
        track_label="potential_person_track_042",
        start_time=datetime.now(timezone.utc),
    )
    db_session.add_all([media, track])
    db_session.commit()

    # Detection 1
    det1 = Detection(
        media_asset_id=media.id,
        track_id=track.id,
        class_name="potential survivor",
        confidence=0.87,
        bbox_x1=100.0,
        bbox_y1=120.0,
        bbox_x2=140.0,
        bbox_y2=200.0,
        frame_number=15,
        image_reference="uploads/frames/frame_0015.jpg",
        latitude=34.055,
        longitude=-118.245,
        location_source="simulated",
        review_status="PENDING_REVIEW",
    )
    # Detection 2 (same track in subsequent frame)
    det2 = Detection(
        media_asset_id=media.id,
        track_id=track.id,
        class_name="potential survivor",
        confidence=0.91,
        bbox_x1=102.0,
        bbox_y1=122.0,
        bbox_x2=143.0,
        bbox_y2=202.0,
        frame_number=18,
        image_reference="uploads/frames/frame_0018.jpg",
        latitude=34.0551,
        longitude=-118.2451,
        location_source="simulated",
        review_status="PENDING_REVIEW",
    )
    db_session.add_all([det1, det2])
    db_session.commit()

    db_session.refresh(track)
    assert len(track.detections) == 2
    assert track.detections[0].class_name == "potential survivor"
    assert track.detections[0].location_source == "simulated"
    assert track.incident.title == "Earthquake Recon"


def test_priority_assessment_and_resource_recommendation_relationships(db_session: Session):
    incident = Incident(title="Flood Mission Bravo", location_name="Basin Zone")
    db_session.add(incident)
    db_session.commit()

    media = MediaAsset(
        incident_id=incident.id,
        file_path="uploads/flood.jpg",
        media_type="IMAGE",
        mime_type="image/jpeg",
        file_size_bytes=4096,
        width_px=800,
        height_px=600,
    )
    db_session.add(media)
    db_session.commit()

    detection = Detection(
        media_asset_id=media.id,
        class_name="person",
        confidence=0.93,
        bbox_x1=50.0,
        bbox_y1=60.0,
        bbox_x2=90.0,
        bbox_y2=150.0,
        location_source="simulated",
        review_status="PENDING_REVIEW",
    )
    db_session.add(detection)
    db_session.commit()

    assessment = PriorityAssessment(
        detection_id=detection.id,
        urgency_score=0.90,
        uncertainty_score=0.10,
        priority_level="CRITICAL_REVIEW",
        rationale="Person detected adjacent to rising flood waters",
    )
    resource = Resource(
        name="Zodiac Inflatable Boat 1",
        resource_type="WATER_RESCUE_BOAT",
        total_capacity=5,
        available_capacity=5,
        simulated_lat=34.1,
        simulated_lon=-118.1,
        is_available=True,
    )
    db_session.add_all([assessment, resource])
    db_session.commit()

    recom = ResourceRecommendation(
        detection_id=detection.id,
        resource_id=resource.id,
        suitability_score=0.95,
        rationale="Nearest aquatic rescue resource for flood zone potential survivor",
        status="PENDING_REVIEW",
        requires_human_authorization=True,
    )
    db_session.add(recom)
    db_session.commit()
    db_session.refresh(detection)

    assert len(detection.priority_assessments) == 1
    assert detection.priority_assessments[0].priority_level == "CRITICAL_REVIEW"
    assert len(detection.recommendations) == 1
    assert detection.recommendations[0].requires_human_authorization is True
    assert detection.recommendations[0].resource.name == "Zodiac Inflatable Boat 1"


def test_cascade_delete_incident_removes_assets_and_detections(db_session: Session):
    incident = Incident(title="Temporary Test Incident", location_name="Testing Area")
    db_session.add(incident)
    db_session.commit()

    media = MediaAsset(
        incident_id=incident.id,
        file_path="uploads/temp.jpg",
        media_type="IMAGE",
        mime_type="image/jpeg",
        file_size_bytes=100,
        width_px=100,
        height_px=100,
    )
    db_session.add(media)
    db_session.commit()

    detection = Detection(
        media_asset_id=media.id,
        class_name="person",
        confidence=0.8,
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=10,
        bbox_y2=10,
    )
    db_session.add(detection)
    db_session.commit()

    # Deleting incident should cascade delete media and detections
    db_session.delete(incident)
    db_session.commit()

    assert db_session.query(Incident).count() == 0
    assert db_session.query(MediaAsset).count() == 0
    assert db_session.query(Detection).count() == 0
