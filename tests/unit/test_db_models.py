"""
Unit tests for SQLAlchemy models, relationships, and persistence.
"""
from sqlalchemy.orm import Session
from app.models.detection import Detection
from app.models.frame import AerialFrame
from app.models.mission import Mission
from app.models.recommendation import ResourceRecommendation
from app.models.resource import Resource
from app.models.review_log import HumanReviewLog


def test_create_mission(db_session: Session):
    mission = Mission(
        title="Earthquake Response Flight 1",
        disaster_type="EARTHQUAKE",
        location_name="District 4",
        simulated_base_lat=34.05,
        simulated_base_lon=-118.25,
        notes="Surveying collapsed infrastructure",
    )
    db_session.add(mission)
    db_session.commit()
    db_session.refresh(mission)

    assert mission.id is not None
    assert mission.is_active is True
    assert mission.created_at is not None


def test_create_frame_and_detection(db_session: Session):
    mission = Mission(
        title="Wildfire Boundary Recon",
        disaster_type="WILDFIRE",
        location_name="Ridge Zone",
        simulated_base_lat=34.10,
        simulated_base_lon=-118.30,
    )
    db_session.add(mission)
    db_session.commit()

    frame = AerialFrame(
        mission_id=mission.id,
        frame_index=1,
        file_path="uploads/frame_001.jpg",
        width_px=1920,
        height_px=1080,
        altitude_m=60.0,
        gimbal_pitch_deg=-90.0,
        is_synthetic=True,
    )
    db_session.add(frame)
    db_session.commit()

    detection = Detection(
        frame_id=frame.id,
        class_name="person",
        confidence=0.91,
        bbox_x1=100.0,
        bbox_y1=150.0,
        bbox_x2=160.0,
        bbox_y2=280.0,
        tracking_id=1,
        estimated_lat=34.1005,
        estimated_lon=-118.3002,
        uncertainty_radius_m=8.5,
        urgency_score=0.85,
        uncertainty_score=0.15,
        priority_level="HIGH",
        review_status="PENDING_REVIEW",
    )
    db_session.add(detection)
    db_session.commit()
    db_session.refresh(detection)

    assert detection.id is not None
    assert detection.frame.mission.title == "Wildfire Boundary Recon"
    assert detection.bbox == {"x1": 100.0, "y1": 150.0, "x2": 160.0, "y2": 280.0}


def test_resource_and_advisory_recommendation(db_session: Session):
    mission = Mission(
        title="Hurricane Surge Mission",
        disaster_type="HURRICANE",
        location_name="Coastal Sector",
        simulated_base_lat=25.76,
        simulated_base_lon=-80.19,
    )
    db_session.add(mission)
    db_session.commit()

    frame = AerialFrame(
        mission_id=mission.id,
        frame_index=1,
        file_path="uploads/coastal_01.jpg",
        width_px=1920,
        height_px=1080,
    )
    db_session.add(frame)
    db_session.commit()

    detection = Detection(
        frame_id=frame.id,
        class_name="potential survivor",
        confidence=0.82,
        bbox_x1=50.0,
        bbox_y1=50.0,
        bbox_x2=120.0,
        bbox_y2=180.0,
        priority_level="CRITICAL_REVIEW",
    )
    db_session.add(detection)

    resource = Resource(
        name="Paramedic Squad 1",
        resource_type="MEDICAL_FIRST_RESPONDER",
        total_capacity=4,
        available_capacity=4,
        simulated_lat=25.765,
        simulated_lon=-80.195,
        is_available=True,
    )
    db_session.add(resource)
    db_session.commit()

    recom = ResourceRecommendation(
        detection_id=detection.id,
        resource_id=resource.id,
        suitability_score=0.92,
        rationale="Critical priority detection within 600m of Paramedic Squad 1.",
        estimated_distance_m=580.0,
        status="PENDING_REVIEW",
        requires_human_authorization=True,
    )
    db_session.add(recom)
    db_session.commit()
    db_session.refresh(recom)

    assert recom.id is not None
    assert recom.requires_human_authorization is True
    assert recom.resource.name == "Paramedic Squad 1"
    assert recom.detection.class_name == "potential survivor"


def test_human_review_log_audit_trail(db_session: Session):
    mission = Mission(
        title="Flood Recon Alpha",
        disaster_type="FLOOD",
        location_name="Zone B",
        simulated_base_lat=30.0,
        simulated_base_lon=31.0,
    )
    db_session.add(mission)
    db_session.commit()

    frame = AerialFrame(
        mission_id=mission.id,
        frame_index=1,
        file_path="uploads/flood_01.jpg",
        width_px=1280,
        height_px=720,
    )
    db_session.add(frame)
    db_session.commit()

    detection = Detection(
        frame_id=frame.id,
        class_name="person",
        confidence=0.79,
        bbox_x1=20.0,
        bbox_y1=30.0,
        bbox_x2=60.0,
        bbox_y2=90.0,
        priority_level="HIGH",
        review_status="PENDING_REVIEW",
    )
    db_session.add(detection)
    db_session.commit()

    # Operator reviews detection
    review_log = HumanReviewLog(
        detection_id=detection.id,
        reviewer_name="Commander Davis",
        action="CONFIRMED",
        notes="Verified potential survivor on partially submerged vehicle.",
    )
    detection.review_status = "CONFIRMED"
    detection.reviewed_by = "Commander Davis"
    db_session.add(review_log)
    db_session.commit()

    db_session.refresh(detection)
    assert detection.review_status == "CONFIRMED"
    assert detection.reviewed_by == "Commander Davis"
    assert len(detection.review_logs) == 1
    assert detection.review_logs[0].action == "CONFIRMED"
