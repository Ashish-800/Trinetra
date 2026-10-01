"""
Command-Line Entry Point for Phase 11 Reproducible Evaluation Suite.
Usage:
  python evaluate.py --all
  python evaluate.py --detection
  python evaluate.py --tracking
  python evaluate.py --workflow
  python evaluate.py --compare
  python evaluate.py --json report.json
"""
import argparse
from pathlib import Path
from app.evaluation.runner import (
    evaluate_decision_workflow,
    evaluate_person_detection,
    evaluate_tracking,
    run_full_evaluation_suite,
)


def main():
    parser = argparse.ArgumentParser(description="AI-Enabled UAV Disaster Assessment - Evaluation Suite")
    parser.add_argument("--all", action="store_true", help="Run full evaluation suite across all modules")
    parser.add_argument("--detection", action="store_true", help="Run person detection evaluation on test set")
    parser.add_argument("--tracking", action="store_true", help="Run multi-object tracking consistency evaluation")
    parser.add_argument("--workflow", action="store_true", help="Run decision workflow and constraint compliance test")
    parser.add_argument("--compare", action="store_true", help="Print Baseline vs Proposed comparative matrix")
    parser.add_argument("--json", type=str, help="Save evaluation results to structured JSON file")
    parser.add_argument("--samples", type=int, default=15, help="Number of VisDrone sample images to evaluate")

    args = parser.parse_args()

    # Default to running all if no specific flag passed
    if not (args.detection or args.tracking or args.workflow or args.compare):
        args.all = True

    if args.all:
        output_path = Path(args.json) if args.json else None
        run_full_evaluation_suite(output_json=output_path)
    else:
        if args.detection:
            m = evaluate_person_detection(max_samples=args.samples)
            print(f"Detection Results: P={m.precision}, R={m.recall}, F1={m.f1_score}, AP@50={m.average_precision_50}, Latency={m.mean_latency_ms}ms")
        if args.tracking:
            t = evaluate_tracking()
            print(f"Tracking Results: Consistency={t.track_consistency*100:.1f}%, Alert Reduction={t.duplicate_alert_reduction_pct}%, ID Switches={t.id_switches}")
        if args.workflow or args.compare:
            w = evaluate_decision_workflow()
            print(f"Workflow Results: Ordering Accuracy={w.priority_ordering_accuracy_pct}%, Availability Compliance={w.availability_compliance_pct}%, Capability Compliance={w.capability_compliance_pct}%")


if __name__ == "__main__":
    main()
