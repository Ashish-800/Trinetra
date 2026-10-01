"""
Unit and integration tests for RescueNet semantic segmentation pipeline.
Verifies dataset loading, image-mask pairing, 12-class model forward pass,
tensor shapes, mask integer preservation, and CPU/CUDA device handling.
"""
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.core.config import settings
from app.schemas.priority import HazardSeverity
from app.services.rescuenet import RescueNetService
from app.services.rescuenet_segmentor import (
    DEFAULT_TARGET_SIZE,
    NUM_RESCUENET_CLASSES,
    RescueNetDataset,
    RescueNetSegmentor,
    create_segmentation_model,
)


@pytest.fixture
def rescuenet_available() -> bool:
    val_dir = settings.BASE_DIR / "RescueNet" / "val"
    return val_dir.exists()


def test_rescuenet_dataset_pairing_and_shapes(rescuenet_available: bool):
    """
    Verifies that RescueNetDataset correctly pairs images with masks,
    resizes both to (512, 512), normalizes images, and preserves discrete mask labels.
    """
    if not rescuenet_available:
        pytest.skip("RescueNet dataset not available")

    dataset = RescueNetDataset(split="val", img_size=(512, 512), max_samples=3)
    assert len(dataset) == 3

    img_tensor, mask_tensor = dataset[0]

    # Verify tensor dimensions and types
    assert img_tensor.shape == (3, 512, 512)
    assert img_tensor.dtype == torch.float32

    assert mask_tensor.shape == (512, 512)
    assert mask_tensor.dtype == torch.int64

    # Verify mask label bounds [0, 11]
    assert mask_tensor.min().item() >= 0
    assert mask_tensor.max().item() <= 11


def test_segmentation_model_instantiation_and_classes():
    """
    Verifies that create_segmentation_model initializes DeepLabV3-MobileNetV3-Large
    with exactly 12 output classes and runs cleanly on CPU/offline.
    """
    model = create_segmentation_model(
        num_classes=NUM_RESCUENET_CLASSES,
        architecture="deeplabv3_mobilenet_v3_large",
        pretrained_backbone=False,
    )
    assert isinstance(model, nn.Module)

    dummy_input = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)

    assert "out" in out
    logits = out["out"]
    # Verify shape: [Batch=1, Classes=12, H=256, W=256]
    assert logits.shape == (1, 12, 256, 256)


def test_segmentor_device_selection():
    """
    Verifies that RescueNetSegmentor respects explicit CPU/CUDA device specifications
    and selects the preferred device correctly.
    """
    # Test explicit CPU device
    segmentor_cpu = RescueNetSegmentor(
        model=create_segmentation_model(num_classes=12, pretrained_backbone=False),
        device="cpu",
    )
    assert segmentor_cpu.device == torch.device("cpu")

    # If CUDA is available, verify CUDA device selection
    if torch.cuda.is_available():
        segmentor_cuda = RescueNetSegmentor(
            model=create_segmentation_model(num_classes=12, pretrained_backbone=False),
            device="cuda:0",
        )
        assert segmentor_cuda.device == torch.device("cuda:0")


def test_segmentor_predict_mask_and_scene_analysis(rescuenet_available: bool):
    """
    Verifies that RescueNetSegmentor runs end-to-end inference on a real image
    and returns a 2D integer mask [512, 512] with labels in [0, 11]
    and transforms it into a structured RescueNetSceneAnalysis object.
    """
    if not rescuenet_available:
        pytest.skip("RescueNet dataset not available")

    img_path = settings.BASE_DIR / "RescueNet" / "val" / "val-org-img" / "10781.jpg"
    assert img_path.exists()

    model = create_segmentation_model(num_classes=12, pretrained_backbone=False)
    segmentor = RescueNetSegmentor(model=model, device="cpu", img_size=(256, 256))

    # Test predict_mask
    pred_mask = segmentor.predict_mask(img_path)
    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == (256, 256)
    assert pred_mask.min() >= 0
    assert pred_mask.max() <= 11

    # Test predict_scene_analysis
    scene_analysis = segmentor.predict_scene_analysis(img_path, image_id="10781")
    assert scene_analysis.image_id == "10781"
    assert scene_analysis.total_pixels == 256 * 256
    assert len(scene_analysis.class_distribution_pct) == 12
    assert scene_analysis.inferred_hazard_severity in list(HazardSeverity)
    assert scene_analysis.inferred_accessibility in ("accessible", "difficult", "isolated")


def test_analyze_mask_array_direct():
    """
    Verifies that RescueNetService.analyze_mask_array correctly translates an in-memory
    numpy array or PyTorch tensor into a complete RescueNetSceneAnalysis with 12 classes.
    """
    service = RescueNetService()

    # Synthetic 100x100 mask:
    # 50% Debris (Class 1) -> triggers CRITICAL hazard and isolated accessibility
    # 30% Road (Class 8)
    # 20% Tree (Class 9)
    mask = np.full((100, 100), fill_value=8, dtype=np.int64)
    mask[:50, :] = 1  # 50% Debris
    mask[80:, :] = 9  # 20% Tree

    analysis = service.analyze_mask_array(mask, image_id="synthetic_test")
    assert analysis.image_id == "synthetic_test"
    assert analysis.total_pixels == 10000
    assert analysis.debris_coverage_pct == 50.0
    assert analysis.road_coverage_pct == 30.0
    assert analysis.tree_coverage_pct == 20.0
    assert analysis.inferred_hazard_severity == HazardSeverity.HIGH
    assert analysis.inferred_accessibility == "isolated"
    assert "HEAVY_DEBRIS_CLEARING" in analysis.inferred_required_capabilities


def test_checkpoint_loading_and_weights_restoration():
    """
    Verifies that RescueNetSegmentor successfully loads the trained prototype checkpoint
    from runs/rescuenet_pilot/best_model.pt (when available) and restores model state.
    """
    ckpt_path = settings.resolved_rescuenet_segmentation_weights_path
    if not ckpt_path.exists():
        pytest.skip(f"Segmentation checkpoint not found at {ckpt_path}")

    segmentor = RescueNetSegmentor(weights_path=ckpt_path, device="cpu")
    assert segmentor.model is not None
    assert segmentor.device == torch.device("cpu")

    # Verify model is in eval mode
    assert not segmentor.model.training


def test_segmentor_predicts_12_classes_within_valid_bounds():
    """
    Verifies that RescueNetSegmentor outputs a 2D integer mask with values strictly
    bounded to [0, 11] corresponding to the 12 RescueNet semantic classes.
    """
    ckpt_path = settings.resolved_rescuenet_segmentation_weights_path
    weights = ckpt_path if ckpt_path.exists() else None

    segmentor = RescueNetSegmentor(weights_path=weights, device="cpu", img_size=(128, 128))
    dummy_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    pred_mask = segmentor.predict_mask(dummy_rgb)
    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == (128, 128)
    assert pred_mask.dtype in (np.int32, np.int64)
    assert pred_mask.min() >= 0
    assert pred_mask.max() <= 11

    scene_analysis = segmentor.predict_scene_analysis(dummy_rgb, image_id="synthetic_12class")
    assert len(scene_analysis.class_distribution_pct) == 12
    # Check that sum of distribution percentages is ~100%
    assert 99.0 <= sum(scene_analysis.class_distribution_pct.values()) <= 101.0


def test_workflow_optional_segmentation_path(db_session, tmp_path):
    """
    Verifies that when enable_segmentation=True and a segmentor is provided,
    the AssessmentWorkflowService executes segmentation inference and populates scene_context.
    """
    import cv2
    from app.services.detector import MockDetector
    from app.services.workflow import AssessmentWorkflowService
    from app.schemas.domain import IncidentCreate

    # Create dummy image
    test_img = tmp_path / "test_aerial.jpg"
    dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(test_img), dummy_img)

    segmentor = RescueNetSegmentor(weights_path=None, device="cpu", img_size=(128, 128))
    workflow = AssessmentWorkflowService(db=db_session, detector=MockDetector(), segmentor=segmentor)

    inc = IncidentCreate(
        title="Optional Segmentation Test Incident",
        disaster_type="FLOOD",
        location_name="Sector 4",
    )

    result = workflow.run_workflow(
        media_path=test_img,
        new_incident=inc,
        enable_segmentation=True,
    )

    assert result.scene_context is not None
    assert result.scene_context.inferred_hazard_severity is not None
    assert result.priority_assessment is not None
    # Verify advisory recommendations retain safety gates
    for rec in result.recommendation_outcome.recommendations:
        assert rec.requires_human_authorization is True

    from app.models.domain import ResourceRecommendation
    recs = db_session.query(ResourceRecommendation).all()
    assert len(recs) > 0
    for r in recs:
        assert r.status == "PENDING_REVIEW"
        assert r.requires_human_authorization is True


def test_workflow_segmentation_disabled_by_default(db_session, tmp_path):
    """
    Verifies that when enable_segmentation is not requested, workflow runs without
    performing segmentation inference, keeping scene_context None.
    """
    import cv2
    from app.services.detector import MockDetector
    from app.services.workflow import AssessmentWorkflowService
    from app.schemas.domain import IncidentCreate

    test_img = tmp_path / "test_aerial_no_seg.jpg"
    dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(test_img), dummy_img)

    segmentor = RescueNetSegmentor(weights_path=None, device="cpu", img_size=(128, 128))
    workflow = AssessmentWorkflowService(db=db_session, detector=MockDetector(), segmentor=segmentor)

    inc = IncidentCreate(
        title="Segmentation Disabled Incident",
        disaster_type="FLOOD",
        location_name="Sector 5",
    )

    result = workflow.run_workflow(
        media_path=test_img,
        new_incident=inc,
        enable_segmentation=False,
    )

    assert result.scene_context is None
    assert result.total_detections_found >= 0


def test_workflow_preserves_ground_truth_mask_priority(db_session, tmp_path):
    """
    Verifies that passing rescuenet_mask_path directly uses ground-truth mask analysis
    even if a segmentation model is configured, preserving existing functionality.
    """
    import cv2
    from app.services.detector import MockDetector
    from app.services.workflow import AssessmentWorkflowService
    from app.schemas.domain import IncidentCreate

    img_path = tmp_path / "dummy_aerial.jpg"
    dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    mask_path = tmp_path / "dummy_mask_lab.png"
    # Ground truth mask filled with water (Class 2)
    dummy_mask = np.full((128, 128), fill_value=2, dtype=np.uint8)
    cv2.imwrite(str(mask_path), dummy_mask)

    segmentor = RescueNetSegmentor(weights_path=None, device="cpu", img_size=(128, 128))
    workflow = AssessmentWorkflowService(db=db_session, detector=MockDetector(), segmentor=segmentor)

    inc = IncidentCreate(
        title="Ground Truth Mask Priority Incident",
        disaster_type="FLOOD",
        location_name="Sector 6",
    )

    result = workflow.run_workflow(
        media_path=img_path,
        new_incident=inc,
        rescuenet_mask_path=mask_path,
        enable_segmentation=True,
    )

    assert result.scene_context is not None
    # Ground truth mask was used: water coverage should be 100%
    assert result.scene_context.water_coverage_pct == 100.0
    assert result.scene_context.image_id == "dummy_mask"


