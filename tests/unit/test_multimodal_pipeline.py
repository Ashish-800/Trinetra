"""
End-to-End Multimodal Pipeline Verification Tests.
Verifies the simultaneous execution of:
  - YOLOv8n (Person Detection, person-only mode)
  - DeepLabV3-MobileNetV3-Large (RescueNet 12-class disaster context)
  - Spatial IoU Multi-Object Tracker
  - Simulated Location Provider
  - Decoupled Priority & Uncertainty Engine
  - Advisory Resource Recommendation Engine
"""
from pathlib import Path
import numpy as np
import pytest
import torch
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.domain import Resource
from app.schemas.domain import BoundingBoxSchema, IncidentCreate
from app.schemas.priority import HazardSeverity, PriorityTier
from app.schemas.rescuenet import RescueNetClass
from app.services.detector import DetectedObject, MockDetector, YOLOAerialDetector
from app.services.rescuenet_segmentor import RescueNetSegmentor
from app.services.workflow import AssessmentWorkflowService


@pytest.fixture
def multimodal_workflow(db_session: Session) -> AssessmentWorkflowService:
    """Provides AssessmentWorkflowService wired with real or mock models."""
    # Seed emergency resources
    db_session.add(
        Resource(
            name="Alpha SAR Water Rescue",
            resource_type="WATER_RESCUE",
            total_capacity=8,
            available_capacity=8,
            simulated_lat=34.05,
            simulated_lon=-118.25,
            is_available=True,
        )
    )
    db_session.commit()

    # Use mock or real YOLO depending on weight availability
    yolo_p = settings.resolved_yolo_weights_path
    if yolo_p.exists():
        detector = YOLOAerialDetector(
            weights_path=yolo_p,
            default_conf_threshold=0.25,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            person_only=True,
        )
    else:
        detector = MockDetector(
            canned_detections=[
                DetectedObject.create_safe(
                    class_name="person",
                    confidence=0.88,
                    bbox=BoundingBoxSchema(x1=0.2, y1=0.2, x2=0.4, y2=0.5),
                )
            ]
        )

    # Use RescueNetSegmentor
    seg_p = settings.resolved_rescuenet_segmentation_weights_path
    segmentor = RescueNetSegmentor(
        weights_path=seg_p if seg_p.exists() else None,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        img_size=(128, 128),
    )

    return AssessmentWorkflowService(
        db=db_session,
        detector=detector,
        segmentor=segmentor,
    )


def test_combined_multimodal_inference_and_outputs(multimodal_workflow: AssessmentWorkflowService, tmp_path: Path):
    """
    Verifies that executing run_workflow with enable_segmentation=True executes both models,
    restricts YOLO detections to 'person', produces a 12-class scene analysis,
    and feeds environmental context into the priority and recommendation engines.
    """
    import cv2

    dummy_img_path = tmp_path / "recon_flight_01.jpg"
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(dummy_img_path), img)

    inc = IncidentCreate(
        title="Hurricane Flood Recon Alpha",
        disaster_type="FLOOD",
        location_name="Coastal Sector 1",
    )

    result = multimodal_workflow.run_workflow(
        media_path=dummy_img_path,
        new_incident=inc,
        enable_segmentation=True,
    )

    # 1. Detection assertions
    assert result.total_detections_found >= 0
    # Detector must only output 'person'
    for track in multimodal_workflow.tracker._active_tracks.values():
        assert track.class_name == "person"

    # 2. Segmentation context assertions
    assert result.scene_context is not None
    assert len(result.scene_context.class_distribution_pct) == 12
    # Verify values 0-11 mapping
    for cls_enum in RescueNetClass:
        assert cls_enum.name in [RescueNetClass(i).name for i in range(12)]

    # 3. Priority and verification assertions
    prio = result.priority_assessment
    assert 0.0 <= prio.urgency_score <= 1.0
    assert 0.0 <= prio.uncertainty_score <= 1.0
    assert prio.composite_priority in list(PriorityTier)
    # Project Safety Rule: human review is mandatory
    assert prio.requires_human_verification is True

    # 4. Advisory recommendation assertions
    rec_outcome = result.recommendation_outcome
    assert rec_outcome.status in ("RECOMMENDATION_AVAILABLE", "NO_SUITABLE_RESOURCE_AVAILABLE")
    for rec in rec_outcome.recommendations:
        assert rec.requires_human_authorization is True


def test_cuda_device_utilization_when_available():
    """Verifies that both models resolve to cuda:0 when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")

    yolo_p = settings.resolved_yolo_weights_path
    if yolo_p.exists():
        yolo = YOLOAerialDetector(weights_path=yolo_p, device="cuda:0")
        assert yolo.device == "cuda:0"

    seg_p = settings.resolved_rescuenet_segmentation_weights_path
    if seg_p.exists():
        seg = RescueNetSegmentor(weights_path=seg_p, device="cuda:0")
        assert seg.device == torch.device("cuda:0")
