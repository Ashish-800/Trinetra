"""
Demonstration and Manual Testing Script for Phase 7 Priority & Verification Logic.

Usage:
  python test_priority_demo.py
"""
from app.schemas.priority import AssessmentInput, HazardSeverity
from app.services.priority_engine import PriorityVerificationEngine


def print_assessment(title: str, input_data: AssessmentInput, engine: PriorityVerificationEngine):
    res = engine.assess(input_data)
    print("\n" + "=" * 70)
    print(f"SCENARIO: {title}")
    print("=" * 70)
    print(f"Inputs:")
    print(f"  Tracked Persons:       {input_data.tracked_person_count}")
    print(f"  Hazard Severity:       {input_data.hazard_severity.value}")
    print(f"  Accessible Route:      {input_data.is_accessible}")
    print(f"  Model Confidence:      {input_data.average_detection_confidence}")
    print(f"  Observation Age:       {input_data.observation_age_minutes} min")
    print(f"  Missing Telemetry:     {input_data.has_missing_telemetry}")
    print(f"  Image Degradation:     {input_data.image_quality_degradation}")

    print("\nResults (Decoupled Dimensions):")
    print(f"  * URGENCY:             {res.urgency_score:.3f} [{res.urgency_level}]")
    print(f"  * UNCERTAINTY:         {res.uncertainty_score:.3f} [{res.uncertainty_level}]")
    print(f"  * COMPOSITE PRIORITY:  {res.composite_priority.value}")
    print(f"  * HUMAN REVIEW GATE:   {res.requires_human_verification} (Safety Mandatory)")

    print("\nContributing Factor Explanations:")
    for f in res.contributing_factors:
        print(f"  - {f.factor_name:30}: contrib={f.contribution:6} | {f.impact_description}")

    print("\nDocumented Data Limitations:")
    for lim in res.data_limitations:
        print(f"  ! {lim}")


def main():
    engine = PriorityVerificationEngine()

    # Scenario 1: High Urgency (Cluster trapped by rising flood, egress blocked)
    print_assessment(
        "1. Active Disaster Crisis (High Urgency)",
        AssessmentInput(
            tracked_person_count=3,
            hazard_severity=HazardSeverity.HIGH,
            is_accessible=False,
            average_detection_confidence=0.90,
            observation_age_minutes=5.0,
        ),
        engine,
    )

    # Scenario 2: Low Urgency (Routine survey, clear roads, no hazards)
    print_assessment(
        "2. Routine Baseline Sweep (Low Urgency)",
        AssessmentInput(
            tracked_person_count=0,
            hazard_severity=HazardSeverity.NONE,
            is_accessible=True,
            observation_age_minutes=2.0,
        ),
        engine,
    )

    # Scenario 3: Uncertain Detection (Low confidence in heavy smoke/dust)
    print_assessment(
        "3. Low Visibility & Model Uncertainty (High Uncertainty)",
        AssessmentInput(
            tracked_person_count=1,
            hazard_severity=HazardSeverity.LOW,
            average_detection_confidence=0.38,
            image_quality_degradation=0.75,
            observation_age_minutes=10.0,
        ),
        engine,
    )

    # Scenario 4: Stale Observation (2.5 hours old flight pass)
    print_assessment(
        "4. Stale Unverified Recon Pass (Temporal Uncertainty)",
        AssessmentInput(
            tracked_person_count=2,
            hazard_severity=HazardSeverity.MODERATE,
            average_detection_confidence=0.88,
            observation_age_minutes=150.0,
        ),
        engine,
    )

    # Scenario 5: Missing Inputs (Missing telemetry and unknown road accessibility)
    print_assessment(
        "5. Missing Inputs & Absent Telemetry",
        AssessmentInput(
            tracked_person_count=1,
            hazard_severity=HazardSeverity.NONE,
            is_accessible=None,
            average_detection_confidence=None,
            has_missing_telemetry=True,
        ),
        engine,
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] All Phase 7 scenarios evaluated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
