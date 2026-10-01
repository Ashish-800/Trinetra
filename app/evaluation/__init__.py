"""
Evaluation Tools and Experiment Benchmarks Package.
Provides reproducible evaluation harnesses for:
- Person Detection (Precision, Recall, F1, AP@50, Processing Latency)
- Multi-Object Tracking (Consistency, ID Switches, Duplicate Alert Quantification)
- Decision Workflow (Priority Ordering, Constraint Compliance, Baseline Comparison)
"""
from app.evaluation.detection_evaluator import DetectionEvaluator, DetectionMetrics
from app.evaluation.tracking_evaluator import TrackingEvaluator, TrackingMetrics
from app.evaluation.workflow_evaluator import WorkflowEvaluationResult, WorkflowEvaluator

__all__ = [
    "DetectionEvaluator",
    "DetectionMetrics",
    "TrackingEvaluator",
    "TrackingMetrics",
    "WorkflowEvaluator",
    "WorkflowEvaluationResult",
]
