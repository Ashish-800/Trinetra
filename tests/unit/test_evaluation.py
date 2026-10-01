"""
Unit and Integration Tests for Phase 11 Evaluation Tools.
Tests detection metrics, tracking metrics, workflow scenario ordering, and constraint checks.
"""
import pytest
from app.evaluation.detection_evaluator import DetectionEvaluator, GroundTruthBox
from app.evaluation.tracking_evaluator import GroundTruthTrajectory, TrackingEvaluator
from app.evaluation.workflow_evaluator import WorkflowEvaluator
from app.schemas.domain import BoundingBoxSchema
from app.services.detector import DetectedObject


# --- 1. Detection Evaluator Tests ---

def test_iou_computation():
    """Tests exact geometric IoU calculation."""
    box_a = BoundingBoxSchema(x1=0.0, y1=0.0, x2=10.0, y2=10.0)  # Area 100
    box_b = BoundingBoxSchema(x1=0.0, y1=0.0, x2=10.0, y2=10.0)  # Area 100
    # Perfect overlap
    assert DetectionEvaluator.compute_iou(box_a, box_b) == 1.0

    # Half overlap: intersection area 50, union area 150 -> 50 / 150 = 1/3
    box_c = BoundingBoxSchema(x1=5.0, y1=0.0, x2=15.0, y2=10.0)
    assert pytest.approx(DetectionEvaluator.compute_iou(box_a, box_c), 0.01) == 0.333

    # Zero overlap
    box_d = BoundingBoxSchema(x1=20.0, y1=20.0, x2=30.0, y2=30.0)
    assert DetectionEvaluator.compute_iou(box_a, box_d) == 0.0


def test_detection_metrics_perfect_match():
    """Tests precision, recall, F1, and AP@50 on a perfect prediction match."""
    evaluator = DetectionEvaluator(iou_threshold=0.50)
    gt_dict = {
        "img1": [
            GroundTruthBox(class_name="person", bbox=BoundingBoxSchema(x1=10, y1=10, x2=50, y2=50)),
            GroundTruthBox(class_name="person", bbox=BoundingBoxSchema(x1=60, y1=60, x2=100, y2=100)),
        ]
    }
    preds_dict = {
        "img1": [
            DetectedObject.create_safe("person", 0.95, BoundingBoxSchema(x1=10, y1=10, x2=50, y2=50)),
            DetectedObject.create_safe("person", 0.90, BoundingBoxSchema(x1=60, y1=60, x2=100, y2=100)),
        ]
    }

    metrics = evaluator.evaluate_predictions(gt_dict, preds_dict, confidence_threshold=0.25)
    assert metrics.status == "evaluated"
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.average_precision_50 == 1.0


def test_detection_metrics_false_positive_and_negative():
    """Tests evaluation when false alarms and missed targets occur."""
    evaluator = DetectionEvaluator(iou_threshold=0.50)
    gt_dict = {
        "img1": [
            GroundTruthBox(class_name="person", bbox=BoundingBoxSchema(x1=10, y1=10, x2=50, y2=50)),  # matched
            GroundTruthBox(class_name="person", bbox=BoundingBoxSchema(x1=60, y1=60, x2=100, y2=100)), # missed -> FN
        ]
    }
    preds_dict = {
        "img1": [
            DetectedObject.create_safe("person", 0.90, BoundingBoxSchema(x1=10, y1=10, x2=50, y2=50)),  # TP
            DetectedObject.create_safe("person", 0.85, BoundingBoxSchema(x1=300, y1=300, x2=350, y2=350)), # FP
        ]
    }

    metrics = evaluator.evaluate_predictions(gt_dict, preds_dict, confidence_threshold=0.25)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5


def test_detection_metrics_missing_ground_truth_never_fabricates():
    """Verifies that empty ground truth returns an honest diagnostic status without fabricated metrics."""
    evaluator = DetectionEvaluator()
    metrics = evaluator.evaluate_predictions(ground_truth_by_image={}, predictions_by_image={})
    assert "metric_cannot_yet_be_calculated" in metrics.status
    assert metrics.precision is None
    assert metrics.recall is None


# --- 2. Tracking Evaluator Tests ---

def test_tracking_evaluation_reduces_duplicate_alerts():
    """Verifies that tracking deduplication dramatically reduces duplicate alert count over raw per-frame detection."""
    evaluator = TrackingEvaluator()
    frames_dets, gt_trajectories = evaluator.generate_synthetic_benchmark_sequence(num_frames=20, num_targets=1)

    metrics = evaluator.evaluate_sequence(frames_dets, gt_trajectories)
    assert metrics.status == "evaluated"
    assert metrics.total_frames_evaluated == 20
    # Naive per-frame fires 1 alert per frame = 20 alerts
    assert metrics.raw_per_frame_detection_alerts == 20
    # Proposed tracking collapses them into 1 persistent track ID
    assert metrics.tracking_deduplicated_alerts == 1
    # Alert reduction must be >= 90%
    assert metrics.duplicate_alert_reduction_pct == 95.0
    assert metrics.track_consistency == 1.0
    assert metrics.id_switches == 0


def test_tracking_evaluation_empty_sequence():
    """Tests handling of empty tracking sequence."""
    evaluator = TrackingEvaluator()
    metrics = evaluator.evaluate_sequence(frames_detections=[], ground_truth_trajectories=[])
    assert "metric_cannot_yet_be_calculated" in metrics.status
    assert metrics.total_frames_evaluated == 0


# --- 3. Decision Workflow Evaluator Tests ---

def test_decision_workflow_priority_ordering():
    """Tests that labeled disaster scenarios are strictly prioritized in the expected order."""
    evaluator = WorkflowEvaluator()
    result = evaluator.evaluate_workflow()

    assert result.status == "evaluated"
    assert result.scenarios_evaluated_count == 4
    # All scenario pairs must respect expected order: Critical > High > Medium > Low
    assert result.priority_ordering_accuracy_pct == 100.0


def test_decision_workflow_constraint_compliance():
    """Verifies that 100% of generated recommendations obey availability, capability, and human review gates."""
    evaluator = WorkflowEvaluator()
    result = evaluator.evaluate_workflow()

    # Unavailable resources must NEVER be recommended
    assert result.availability_compliance_pct == 100.0
    # Mismatched capabilities must NEVER be recommended
    assert result.capability_compliance_pct == 100.0
    # Every recommendation must have requires_human_authorization = True
    assert result.human_authorization_compliance_pct == 100.0


def test_baseline_vs_proposed_comparison_content():
    """Tests that the comparative ablation covers fatigue, priority, resource allocation, and safety."""
    evaluator = WorkflowEvaluator()
    result = evaluator.evaluate_workflow()

    dimensions = [comp.metric_name for comp in result.baseline_comparisons]
    assert any("Alert Fatigue" in d for d in dimensions)
    assert any("Priority Evaluation" in d for d in dimensions)
    assert any("Resource Allocation" in d for d in dimensions)
    assert any("Safety" in d for d in dimensions)
