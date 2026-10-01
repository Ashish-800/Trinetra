"""
Integration tests for AssessmentWorkflowService (Phase 9).
Tests the controlled 8-step pipeline using mocked model output and simulated data.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.models.domain import (
    Detection,
    Incident,
    MediaAsset,
    PriorityAssessment,
    ResourceRecommendation,
    Track,
)
from app.schemas.domain import BoundingBoxSchema, IncidentCreate
from app.schemas.location import DroneTelemetryMetadata, LocationSource
from app.schemas.priority import HazardSeverity
from app.services.detector import DetectedObject, MockDetector
from app.services.location import SimulatedLocationProvider
from app.services.workflow import AssessmentWorkflowService, IncidentNotFoundError


@pytest.fixture
def synthetic_test_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "flood_recon.jpg"
    img = np.full((480, 640, 3), (110, 140, 100), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def synthetic_test_video(tmp_path: Path) -> Path:
    vid_path = tmp_path / "uav_flight_sweep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))
    for i in range(10):
        frame = np.full((64, 64, 3), 100 + i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()
    return vid_path


def test_workflow_execution_with_image(db_session: Session, synthetic_test_image: Path):
    """
    Tests complete workflow on a single aerial image:
    1. Create incident
    2. Register image media
    3. Run detection (MockDetector with 1 person)
    4. Track detections (1 static track)
    5. Attach simulated location
    6. Calculate urgency & verification status
    7. Generate resource recommendations
    8. Save all results in DB
    """
    mock_objs = [
        DetectedObject.create_safe("person", 0.92, BoundingBoxSchema(x1=200, y1=150, x2=250, y2=280))
    ]
    detector = MockDetector(mock_objects=mock_objs)
    workflow = AssessmentWorkflowService(db=db_session, detector=detector)

    new_inc = IncidentCreate(
        title="Hurricane Surge Sector 1",
        disaster_type="FLOOD",
        location_name="Coastal Marina",
    )

    result = workflow.run_workflow(
        media_path=synthetic_test_image,
        new_incident=new_inc,
        hazard_severity=HazardSeverity.HIGH,
        is_accessible=False,
    )

    # Validate returned summary
    assert result.incident_id is not None
    assert result.incident_title == "Hurricane Surge Sector 1"
    assert result.media_type == "IMAGE"
    assert result.frames_processed == 1
    assert result.total_detections_found == 1
    assert result.unique_person_tracks == 1
    assert result.locations_attached_count == 1
    assert result.priority_assessment.composite_priority.value in ("CRITICAL_REVIEW", "HIGH")
    assert result.recommendation_outcome.status == "RECOMMENDATION_AVAILABLE"
    assert len(result.recommendation_outcome.recommendations) >= 1

    # Verify Step 8 Database Persistence
    db_incident = db_session.query(Incident).filter(Incident.id == result.incident_id).first()
    assert db_incident is not None
    assert len(db_incident.media_assets) == 1

    media_asset = db_incident.media_assets[0]
    assert media_asset.media_type == "IMAGE"
    assert len(media_asset.detections) == 1

    det = media_asset.detections[0]
    assert det.class_name == "person"
    assert det.latitude is not None
    assert det.longitude is not None
    assert det.location_source == "simulated"

    # Verify Priority Assessment saved in DB
    assessments = db_session.query(PriorityAssessment).filter(PriorityAssessment.detection_id == det.id).all()
    assert len(assessments) == 1
    assert assessments[0].urgency_score > 0.50

    # Verify Resource Recommendation saved in DB
    recoms = db_session.query(ResourceRecommendation).filter(ResourceRecommendation.detection_id == det.id).all()
    assert len(recoms) >= 1
    assert recoms[0].requires_human_authorization is True


def test_workflow_execution_with_video(db_session: Session, synthetic_test_video: Path):
    """
    Tests complete workflow on a UAV video:
    Verifies multi-frame extraction, tracking consistency, and deduplication.
    """
    mock_objs = [
        DetectedObject.create_safe("person", 0.88, BoundingBoxSchema(x1=20, y1=20, x2=40, y2=50))
    ]
    detector = MockDetector(mock_objects=mock_objs)
    workflow = AssessmentWorkflowService(db=db_session, detector=detector)

    result = workflow.run_workflow(
        media_path=synthetic_test_video,
        sampling_interval_seconds=0.5,  # Extracts 2 frames
        hazard_severity=HazardSeverity.MODERATE,
    )

    assert result.media_type == "VIDEO"
    assert result.frames_processed >= 2
    assert result.total_detections_found >= 2
    # Tracking deduplication: 1 unique person across all frames!
    assert result.unique_person_tracks == 1

    # Check that Track records were persisted in DB
    db_tracks = db_session.query(Track).filter(Track.incident_id == result.incident_id).all()
    assert len(db_tracks) == 1
    assert "person_track" in db_tracks[0].track_label


def test_workflow_with_existing_incident_id(db_session: Session, synthetic_test_image: Path):
    """Tests associating new media with a pre-existing incident."""
    existing_incident = Incident(
        title="Pre-Existing Incident",
        disaster_type="EARTHQUAKE",
        location_name="Downtown Plaza",
    )
    db_session.add(existing_incident)
    db_session.commit()
    db_session.refresh(existing_incident)

    workflow = AssessmentWorkflowService(db=db_session)
    result = workflow.run_workflow(
        media_path=synthetic_test_image,
        incident_id=existing_incident.id,
    )

    assert result.incident_id == existing_incident.id
    assert result.incident_title == "Pre-Existing Incident"


def test_workflow_with_invalid_incident_id_raises_error(db_session: Session, synthetic_test_image: Path):
    """Verifies that referencing a non-existent incident ID raises IncidentNotFoundError."""
    workflow = AssessmentWorkflowService(db=db_session)
    with pytest.raises(IncidentNotFoundError) as exc_info:
        workflow.run_workflow(
            media_path=synthetic_test_image,
            incident_id=99999,  # Non-existent ID!
        )
    assert "not found in database" in str(exc_info.value)


def test_end_to_end_pipeline_with_trained_yolo_detector(db_session: Session):
    """
    Tests the complete end-to-end production pipeline with the trained YOLO detector:
    Image -> YOLO Detection -> Tracking -> Location Estimation -> Priority -> Recommendations -> DB.
    """
    from app.core.config import settings
    from app.services.detector import YOLOAerialDetector

    trained_weights = settings.resolved_yolo_weights_path
    if not trained_weights.exists():
        pytest.skip(f"Trained weights not found at {trained_weights}")

    test_image = Path("archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg")
    if not test_image.exists():
        pytest.skip(f"Validation image not found at {test_image}")

    detector = YOLOAerialDetector(
        weights_path=trained_weights,
        default_conf_threshold=0.35,
        person_only=True,
    )
    workflow = AssessmentWorkflowService(db=db_session, detector=detector)

    new_inc = IncidentCreate(
        title="VisDrone UAV Disaster Assessment Survey",
        disaster_type="FLOOD",
        location_name="Urban Disaster Zone Alpha",
        notes="Reconnaissance for potential survivors",
    )

    result = workflow.run_workflow(
        media_path=test_image,
        new_incident=new_inc,
        hazard_severity=HazardSeverity.HIGH,
        is_accessible=False,
    )

    # 1. Pipeline execution result validation
    assert result.incident_id is not None
    assert result.media_asset_id is not None
    assert result.frames_processed == 1
    assert result.total_detections_found > 0
    assert result.unique_person_tracks > 0
    assert result.locations_attached_count == result.total_detections_found
    assert result.priority_assessment is not None
    assert result.recommendation_outcome is not None

    # 2. Database persistence verification
    db_dets = db_session.query(Detection).filter(Detection.media_asset_id == result.media_asset_id).all()
    assert len(db_dets) == result.total_detections_found
    for det in db_dets:
        assert det.class_name in ("person", "potential survivor")
        assert det.location_source == "simulated"
        assert det.latitude is not None
        assert det.longitude is not None
        assert det.uncertainty_radius_m is not None
        assert det.review_status == "PENDING_REVIEW"

    # Tracks: check that tracks are created
    db_tracks = db_session.query(Track).filter(Track.incident_id == result.incident_id).all()
    assert len(db_tracks) == result.unique_person_tracks

    # Priority Assessments: check that each detection has a linked PriorityAssessment
    db_assessments = db_session.query(PriorityAssessment).all()
    assert len(db_assessments) == len(db_dets)
    for pa in db_assessments:
        assert pa.priority_level in ("CRITICAL_REVIEW", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL")
        assert pa.urgency_score >= 0.0
        assert pa.uncertainty_score >= 0.0

    # Resource Recommendations: check that recommendations require human auth
    db_recs = db_session.query(ResourceRecommendation).all()
    assert len(db_recs) > 0
    for r in db_recs:
        assert r.requires_human_authorization is True
        assert r.status == "PENDING_REVIEW"


def test_end_to_end_video_pipeline_with_trained_yolo_detector(db_session: Session, tmp_path: Path):
    """
    Tests the complete end-to-end video pipeline with the trained YOLO detector:
    Video -> Frame Extraction -> YOLO Detection -> Multi-Frame Tracking ->
    Simulated Location -> Urgency/Priority -> Resource Recommendation -> DB.
    """
    from collections import Counter
    from app.core.config import settings
    from app.services.detector import YOLOAerialDetector

    trained_weights = settings.resolved_yolo_weights_path
    if not trained_weights.exists():
        pytest.skip(f"Trained weights not found at {trained_weights}")

    base_image_path = Path("archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg")
    if not base_image_path.exists():
        pytest.skip(f"Validation image not found at {base_image_path}")

    # Create a small 3-frame video with minor shift to verify tracking persistence
    vid_path = tmp_path / "uav_test_flight.mp4"
    img = cv2.imread(str(base_image_path))
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 2.0, (w, h))
    for i in range(3):
        M = np.float32([[1, 0, i * 2], [0, 1, i * 1]])
        shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.write(shifted)
    out.release()

    detector = YOLOAerialDetector(
        weights_path=trained_weights,
        default_conf_threshold=0.35,
        person_only=True,
    )
    workflow = AssessmentWorkflowService(db=db_session, detector=detector)

    result = workflow.run_workflow(
        media_path=vid_path,
        sampling_interval_seconds=0.5,
        hazard_severity=HazardSeverity.HIGH,
        is_accessible=False,
    )

    # 1. Pipeline execution result verification
    assert result.media_type == "VIDEO"
    assert result.frames_processed == 3
    assert result.total_detections_found > 0
    # Multi-frame tracking deduplication: unique tracks must be fewer than total detections across frames
    assert result.unique_person_tracks < result.total_detections_found
    assert result.unique_person_tracks > 0
    assert result.locations_attached_count == result.total_detections_found

    # 2. Database verification: check that multiple detections share the same track_id
    db_dets = db_session.query(Detection).filter(Detection.media_asset_id == result.media_asset_id).all()
    assert len(db_dets) == result.total_detections_found

    track_counts = Counter(d.track_id for d in db_dets)
    persisting_tracks = [tid for tid, count in track_counts.items() if count > 1]
    assert len(persisting_tracks) > 0, "Expected at least one track to persist across consecutive video frames"

    # All detections must be 'person' and 'simulated'
    for d in db_dets:
        assert d.class_name in ("person", "potential survivor")
        assert d.location_source == "simulated"
        assert d.review_status == "PENDING_REVIEW"

    # Priority & Recommender verification
    assert result.priority_assessment.requires_human_verification is True
    assert len(result.recommendation_outcome.recommendations) > 0
    for rec in result.recommendation_outcome.recommendations:
        assert rec.requires_human_authorization is True


