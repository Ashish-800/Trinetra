"""
Integration and Domain Tests for RescueNet Disaster Scene Context Layer.
Verifies dataset loading, mask compatibility, transparent hazard inference,
and integration into the assessment workflow while preserving the person-detection pipeline.
"""
from pathlib import Path
import cv2
import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.domain import Detection, PriorityAssessment, ResourceRecommendation
from app.schemas.domain import IncidentCreate
from app.schemas.priority import HazardSeverity, PriorityTier
from app.schemas.rescuenet import (
    CLASS_DISPLAY_NAMES,
    RescueNetClass,
    RescueNetHazardThresholds,
    RescueNetSceneAnalysis,
)
from app.services.detector import MockDetector, YOLOAerialDetector
from app.services.rescuenet import RescueNetDatasetError, RescueNetService
from app.services.workflow import AssessmentWorkflowService


@pytest.fixture
def sample_rescuenet_pair() -> tuple[Path, Path]:
    """Provides a verified real RescueNet validation image and mask pair."""
    img = settings.BASE_DIR / "RescueNet" / "val" / "val-org-img" / "10781.jpg"
    mask = settings.BASE_DIR / "RescueNet" / "val" / "val-label-img" / "10781_lab.png"
    if not img.exists() or not mask.exists():
        pytest.skip(f"RescueNet validation samples not found at {img}")
    return img, mask


def test_rescuenet_dataset_availability_and_loading():
    """Verifies RescueNet dataset directory presence and sample pair discovery."""
    svc = RescueNetService()
    assert svc.is_available() is True

    samples = svc.list_samples(split="val", max_count=5)
    assert len(samples) > 0
    for img_path, mask_path in samples:
        assert img_path.exists()
        assert mask_path.exists()
        assert mask_path.name == f"{img_path.stem}_lab.png"


def test_rescuenet_mask_image_compatibility(sample_rescuenet_pair: tuple[Path, Path]):
    """Verifies that RescueNet image and mask dimensions match and analyze_sample discovers the mask."""
    img_path, mask_path = sample_rescuenet_pair
    svc = RescueNetService()

    # Image and mask dimensional compatibility
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None and mask is not None
    assert img.shape[:2] == mask.shape[:2]

    # Automatic mask resolution via analyze_sample
    analysis = svc.analyze_sample(image_path=img_path)
    assert analysis.image_id == "10781"
    assert analysis.total_pixels == img.shape[0] * img.shape[1]


def test_rescuenet_hazard_context_generation_and_thresholds(sample_rescuenet_pair: tuple[Path, Path]):
    """
    Verifies that RescueNet produces structured disaster scene context
    without claiming person/survivor detection, using transparent configurable thresholds.
    """
    _, mask_path = sample_rescuenet_pair
    custom_thresh = RescueNetHazardThresholds(
        critical_destroyed_building_pct=0.5,
        critical_water_pct=20.0,
    )
    svc = RescueNetService(thresholds=custom_thresh)
    analysis = svc.analyze_scene(mask_path)

    # 1. Macro environmental context fields
    assert analysis.total_pixels > 0
    assert 0.0 <= analysis.water_coverage_pct <= 100.0
    assert 0.0 <= analysis.debris_coverage_pct <= 100.0
    assert 0.0 <= analysis.destroyed_building_pct <= 100.0
    assert analysis.inferred_hazard_severity in list(HazardSeverity)
    assert analysis.inferred_accessibility in ("accessible", "difficult", "isolated", "impassable")
    assert len(analysis.inferred_required_capabilities) > 0
    assert len(analysis.contributing_hazard_factors) > 0
    assert analysis.thresholds_used == custom_thresh

    # 2. Safety Rule: RescueNet must NEVER claim survivor or person detection
    class_names = [k.lower() for k in analysis.class_distribution_pct.keys()]
    assert "person" not in class_names
    assert "survivor" not in class_names
    assert not hasattr(analysis, "person_count")
    assert not hasattr(analysis, "survivor_count")


def test_rescuenet_integration_with_workflow(db_session: Session, sample_rescuenet_pair: tuple[Path, Path]):
    """
    Verifies that passing rescuenet_mask_path automatically informs the workflow's
    hazard severity, accessibility, and resource capabilities when not manually overridden.
    """
    img_path, mask_path = sample_rescuenet_pair
    workflow = AssessmentWorkflowService(db=db_session, detector=MockDetector())

    new_inc = IncidentCreate(
        title="Hurricane Michael Sector Reconnaissance",
        disaster_type="FLOOD",
        location_name="Panama City Coastal Sector",
        notes="Automated assessment with RescueNet environmental damage context",
    )

    result = workflow.run_workflow(
        media_path=img_path,
        new_incident=new_inc,
        hazard_severity=HazardSeverity.NONE,  # Unspecified; should be inferred from RescueNet
        is_accessible=None,                  # Unspecified; should be inferred from RescueNet
        rescuenet_mask_path=mask_path,
    )

    # 1. Context attached to execution result
    assert result.scene_context is not None
    assert result.scene_context.image_id == "10781"
    assert result.scene_context.debris_coverage_pct > 10.0

    # 2. Priority Engine received inferred values
    # In 10781, debris is > 40% and destroyed buildings > 1%, inferring CRITICAL and isolated (is_accessible=False)
    assert result.priority_assessment.urgency_score > 0.60
    assert result.priority_assessment.composite_priority == PriorityTier.CRITICAL_REVIEW

    # 3. Advisory resource recommendations generated
    assert len(result.recommendation_outcome.recommendations) > 0
    for rec in result.recommendation_outcome.recommendations:
        assert rec.requires_human_authorization is True


def test_rescuenet_manual_hazard_override_preserved(db_session: Session, sample_rescuenet_pair: tuple[Path, Path]):
    """
    Verifies that explicit operator-specified hazard_severity overrides
    the RescueNet inferred severity, preserving backward compatibility.
    """
    img_path, mask_path = sample_rescuenet_pair
    workflow = AssessmentWorkflowService(db=db_session, detector=MockDetector())

    result = workflow.run_workflow(
        media_path=img_path,
        hazard_severity=HazardSeverity.LOW,  # Explicit operator override
        is_accessible=True,                  # Explicit operator override
        rescuenet_mask_path=mask_path,
    )

    # Operator override takes precedence in PriorityAssessment
    assert result.scene_context is not None
    # RescueNet inferred severity was CRITICAL, but operator override was LOW
    assert result.scene_context.inferred_hazard_severity == HazardSeverity.CRITICAL
    # The actual priority assessment used LOW hazard severity, yielding lower urgency
    assert result.priority_assessment.urgency_score < 0.60


def test_combined_rescuenet_and_visdrone_pipeline(db_session: Session, sample_rescuenet_pair: tuple[Path, Path]):
    """
    Verifies combined dual-layer analysis:
    - Micro layer: Real VisDrone validation aerial image analyzed by trained YOLOv8n detector
    - Macro layer: Real RescueNet disaster mask analyzed for environmental damage context
    """
    _, mask_path = sample_rescuenet_pair
    visdrone_img = Path("archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg")
    if not visdrone_img.exists():
        pytest.skip(f"VisDrone validation image not found at {visdrone_img}")

    trained_weights = settings.resolved_yolo_weights_path
    if not trained_weights.exists():
        pytest.skip(f"Trained weights not found at {trained_weights}")

    detector = YOLOAerialDetector(
        weights_path=trained_weights,
        default_conf_threshold=0.35,
        person_only=True,
    )
    workflow = AssessmentWorkflowService(db=db_session, detector=detector)

    result = workflow.run_workflow(
        media_path=visdrone_img,
        hazard_severity=HazardSeverity.NONE,  # Derived from RescueNet
        is_accessible=None,                  # Derived from RescueNet
        rescuenet_mask_path=mask_path,
    )

    # 1. Micro layer: candidate person detections from YOLO
    assert result.total_detections_found > 0
    assert result.unique_person_tracks > 0
    db_dets = db_session.query(Detection).filter(Detection.media_asset_id == result.media_asset_id).all()
    for d in db_dets:
        assert d.class_name in ("person", "potential survivor")
        assert d.location_source == "simulated"

    # 2. Macro layer: environmental context from RescueNet
    assert result.scene_context is not None
    assert result.scene_context.water_coverage_pct > 0.0
    assert result.scene_context.debris_coverage_pct > 0.0

    # 3. Decision-support fusion: Priority reflects person count + environmental hazard
    assert result.priority_assessment.composite_priority == PriorityTier.CRITICAL_REVIEW
    assert result.priority_assessment.requires_human_verification is True

    # 4. Advisory recommendations require human authorization
    db_recs = db_session.query(ResourceRecommendation).all()
    assert len(db_recs) > 0
    for r in db_recs:
        assert r.requires_human_authorization is True
        assert r.status == "PENDING_REVIEW"


def test_rescuenet_taxonomy_audit():
    """
    Dataset-to-Implementation Audit Test:
    Verifies that the codebase implements all 12 classes (Background + 11 foreground disaster classes)
    matching the official RescueNet v1.0 dataset specification and version note.
    """
    expected_classes = {
        0: "Background",
        1: "Debris",
        2: "Water",
        3: "Building No Damage",
        4: "Building Minor Damage",
        5: "Building Major Damage",
        6: "Building Total Destruction",
        7: "Vehicle",
        8: "Road",
        9: "Tree",
        10: "Pool",
        11: "Sand",
    }

    # Verify enum member count
    assert len(RescueNetClass) == 12

    # Verify ID to name mapping
    for cls_id, cls_name in expected_classes.items():
        enum_val = RescueNetClass(cls_id)
        assert CLASS_DISPLAY_NAMES[enum_val] == cls_name

    # Verify RescueNetSceneAnalysis has dedicated attributes for all environmental classes
    sample_analysis = RescueNetSceneAnalysis(
        image_id="audit_test",
        total_pixels=1000,
        water_coverage_pct=5.0,
        debris_coverage_pct=10.0,
        destroyed_building_pct=2.0,
        major_damage_building_pct=1.0,
        minor_damage_building_pct=0.5,
        intact_building_pct=8.0,
        road_coverage_pct=15.0,
        tree_coverage_pct=25.0,
        pool_coverage_pct=1.5,
        sand_coverage_pct=20.0,
        vehicle_detected=True,
    )
    assert sample_analysis.tree_coverage_pct == 25.0
    assert sample_analysis.pool_coverage_pct == 1.5
    assert sample_analysis.sand_coverage_pct == 20.0


def test_rescuenet_three_real_val_samples():
    """
    Verifies scene analysis on 3 diverse real validation image-mask pairs from RescueNet:
    - 10781: Mixed structural failure, flood water, and debris
    - 10793: Coastal sand terrain with destroyed building and trees
    - 10842: Heavy debris field with swimming pool and vehicle
    """
    service = RescueNetService()
    if not service.is_available():
        pytest.skip("RescueNet dataset not available in local environment")

    val_img_dir = Path("RescueNet/val/val-org-img")
    val_lab_dir = Path("RescueNet/val/val-label-img")

    stems = ["10781", "10793", "10842"]
    for stem in stems:
        img_p = val_img_dir / f"{stem}.jpg"
        lab_p = val_lab_dir / f"{stem}_lab.png"
        assert img_p.exists(), f"Missing image {img_p}"
        assert lab_p.exists(), f"Missing mask {lab_p}"

        analysis = service.analyze_sample(img_p, mask_path=lab_p)
        assert analysis.image_id == stem
        assert analysis.total_pixels > 0
        assert len(analysis.class_distribution_pct) == 12

    # Specific assertions for each diverse sample
    # 10781: High debris, water, multiple building damage states
    a_10781 = service.analyze_sample(val_img_dir / "10781.jpg")
    assert a_10781.water_coverage_pct > 5.0
    assert a_10781.debris_coverage_pct > 40.0
    assert a_10781.inferred_hazard_severity == HazardSeverity.CRITICAL

    # 10793: Coastal sand terrain
    a_10793 = service.analyze_sample(val_img_dir / "10793.jpg")
    assert a_10793.sand_coverage_pct > 70.0
    assert a_10793.destroyed_building_pct > 5.0

    # 10842: Massive debris, vehicle detected, and pool present
    a_10842 = service.analyze_sample(val_img_dir / "10842.jpg")
    assert a_10842.debris_coverage_pct > 70.0
    assert a_10842.pool_coverage_pct > 0.0
    assert a_10842.vehicle_detected is True

