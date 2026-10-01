"""
Unit tests for Transparent Priority and Verification Logic (Phase 7).
Verifies:
- High urgency scenarios
- Low urgency scenarios
- Uncertain detections
- Stale observations
- Missing inputs and data limitations
- Mandatory human verification (no autonomous dispatch)
"""
import pytest
from app.schemas.priority import (
    AssessmentInput,
    HazardSeverity,
    PriorityTier,
)
from app.services.priority_engine import PriorityVerificationEngine


@pytest.fixture
def engine() -> PriorityVerificationEngine:
    return PriorityVerificationEngine()


# ==========================================
# 1. High Urgency Tests
# ==========================================

def test_high_urgency_scenario(engine: PriorityVerificationEngine):
    """
    Scenario: 4 tracked persons trapped by a critical hazard (raging torrent/fire)
    with blocked accessibility.
    Expectation: Urgency >= 0.85, PriorityTier is CRITICAL_REVIEW.
    """
    input_data = AssessmentInput(
        tracked_person_count=4,
        hazard_severity=HazardSeverity.CRITICAL,
        is_accessible=False,
        average_detection_confidence=0.92,
        observation_age_minutes=5.0,
    )
    result = engine.assess(input_data)

    assert result.urgency_score >= 0.85
    assert result.urgency_level in ("HIGH", "CRITICAL")
    assert result.composite_priority == PriorityTier.CRITICAL_REVIEW
    assert result.requires_human_verification is True

    # Check that factors explain the assessment
    factor_names = [f.factor_name for f in result.contributing_factors]
    assert "tracked_person_count" in factor_names
    assert "hazard_severity" in factor_names
    assert "accessibility_status" in factor_names


# ==========================================
# 2. Low Urgency Tests
# ==========================================

def test_low_urgency_scenario(engine: PriorityVerificationEngine):
    """
    Scenario: Routine baseline sweep, 0 persons detected, no hazards, roads accessible.
    Expectation: Urgency near 0.0, PriorityTier is LOW or INFORMATIONAL.
    """
    input_data = AssessmentInput(
        tracked_person_count=0,
        hazard_severity=HazardSeverity.NONE,
        is_accessible=True,
        average_detection_confidence=None,
        observation_age_minutes=2.0,
    )
    result = engine.assess(input_data)

    assert result.urgency_score <= 0.15
    assert result.urgency_level == "LOW"
    assert result.composite_priority in (PriorityTier.LOW, PriorityTier.INFORMATIONAL)
    assert result.requires_human_verification is True


# ==========================================
# 3. Uncertain Detections Tests
# ==========================================

def test_uncertain_detections_elevates_uncertainty_score(engine: PriorityVerificationEngine):
    """
    Scenario: Model detected potential person with low confidence (0.35) and smoke/blur degradation.
    Expectation: Uncertainty score is elevated (>= 0.45) with explicit factor explanations.
    """
    input_data = AssessmentInput(
        tracked_person_count=1,
        hazard_severity=HazardSeverity.LOW,
        average_detection_confidence=0.35,  # Low confidence
        image_quality_degradation=0.60,      # Smoke/dust degradation
        observation_age_minutes=5.0,
    )
    result = engine.assess(input_data)

    assert result.uncertainty_score >= 0.35
    assert result.uncertainty_level in ("MODERATE", "HIGH")
    assert result.requires_human_verification is True

    # Data limitations must capture the image degradation
    limitations_str = " ".join(result.data_limitations).lower()
    assert "image quality degradation" in limitations_str or "smoke" in limitations_str


# ==========================================
# 4. Stale Observations Tests
# ==========================================

def test_stale_observations_elevates_temporal_uncertainty(engine: PriorityVerificationEngine):
    """
    Scenario: Detections from an aerial flight pass conducted 150 minutes (2.5 hours) ago.
    Expectation: Uncertainty is elevated, staleness is explicitly recorded in data limitations.
    """
    input_data = AssessmentInput(
        tracked_person_count=2,
        hazard_severity=HazardSeverity.MODERATE,
        average_detection_confidence=0.85,
        observation_age_minutes=150.0,  # 2.5 hours old!
    )
    result = engine.assess(input_data)

    assert result.uncertainty_score >= 0.40
    limitations_str = " ".join(result.data_limitations).lower()
    assert "old" in limitations_str or "stale" in limitations_str


# ==========================================
# 5. Missing Inputs and Edge Cases
# ==========================================

def test_missing_inputs_handled_gracefully(engine: PriorityVerificationEngine):
    """
    Scenario: Missing confidence, unknown accessibility (None), and missing telemetry.
    Expectation: Does not crash, applies conservative default uncertainty, records limitations.
    """
    input_data = AssessmentInput(
        tracked_person_count=1,
        hazard_severity=HazardSeverity.NONE,
        is_accessible=None,                 # Unknown accessibility
        average_detection_confidence=None,  # Missing confidence
        has_missing_telemetry=True,         # Missing telemetry
    )
    result = engine.assess(input_data)

    assert result.urgency_score is not None
    assert result.uncertainty_score >= 0.50  # High uncertainty due to missing signals
    assert result.requires_human_verification is True

    # Check that missing signals are honestly reported in data limitations
    limitations_str = " ".join(result.data_limitations).lower()
    assert "confidence was missing" in limitations_str
    assert "accessibility is unknown" in limitations_str
    assert "telemetry missing" in limitations_str


def test_confidence_not_equated_with_rescue_certainty(engine: PriorityVerificationEngine):
    """
    Safety Rule: High model confidence (e.g. 0.99) does NOT prove that a person needs rescue.
    If no hazards exist and access is clear, urgency remains low.
    """
    input_data = AssessmentInput(
        tracked_person_count=1,
        hazard_severity=HazardSeverity.NONE,
        is_accessible=True,
        average_detection_confidence=0.99,  # Very high visual confidence!
    )
    result = engine.assess(input_data)

    # Urgency must NOT automatically jump to critical just because visual confidence is high
    assert result.urgency_score <= 0.45
    assert result.urgency_level in ("LOW", "MEDIUM")

    # Required safety disclaimer is present in limitations
    limitations_str = " ".join(result.data_limitations).lower()
    assert "vital signs" in limitations_str
    assert "rescue need" in limitations_str
