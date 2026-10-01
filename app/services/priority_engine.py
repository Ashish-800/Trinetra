"""
Transparent Priority and Verification Scoring Engine.
Keeps Urgency (concern level) and Uncertainty (verification need) strictly separate.
All scoring weights are explicitly labeled as prototype assumptions.
"""
from typing import List
from app.schemas.priority import (
    AssessmentInput,
    FactorExplanation,
    HazardSeverity,
    PriorityAssessmentResult,
    PriorityTier,
)


class PrototypeScoringAssumptions:
    """
    Explicitly documented prototype scoring weights and baseline assumptions.
    These weights are heuristic baseline assumptions for prototype demonstration,
    not empirically calibrated disaster loss curves.
    """
    # Urgency Factor Weights (Sum to 1.0)
    WEIGHT_URGENCY_PERSONS = 0.40      # Assumption: Human presence is the primary driver of rescue concern
    WEIGHT_URGENCY_HAZARDS = 0.40      # Assumption: Visible environmental hazards compound life safety risk
    WEIGHT_URGENCY_INACCESSIBLE = 0.20 # Assumption: Blocked egress prevents self-evacuation

    # Uncertainty Baseline Thresholds
    STALE_OBSERVATION_MINUTES = 60.0   # Assumption: Imagery older than 1 hour has high temporal drift
    LOW_CONFIDENCE_THRESHOLD = 0.50    # Assumption: Detections under 0.50 require substantial verification
    TELEMETRY_MISSING_PENALTY = 0.30   # Assumption: Missing drone telemetry severely limits spatial actionability


class PriorityVerificationEngine:
    """
    Evaluates disaster assessment inputs and outputs decoupled Urgency and Uncertainty scores.
    Enforces Safety Rule: Confidence does NOT prove rescue need; human review is mandatory.
    """

    def __init__(self, assumptions: PrototypeScoringAssumptions = PrototypeScoringAssumptions()):
        self.assumptions = assumptions

    def evaluate(self, input_data: AssessmentInput) -> PriorityAssessmentResult:
        """Alias for assess() supporting benchmark evaluation callers."""
        return self.assess(input_data)

    def assess(self, input_data: AssessmentInput) -> PriorityAssessmentResult:
        factors: List[FactorExplanation] = []
        limitations: List[str] = [
            "Visual imagery alone cannot determine vital signs, consciousness, injury, or entrapment.",
            "Detection confidence reflects visual pattern resemblance, NOT certainty of rescue need.",
        ]

        # -------------------------------------------------------------
        # 1. URGENCY EVALUATION: How concerning is the observed zone?
        # -------------------------------------------------------------

        # Factor 1.1: Distinct Tracked Person Count
        person_count = input_data.tracked_person_count
        if person_count == 0:
            u_person = 0.0
            p_desc = "No distinct persons detected in current sweep."
        elif person_count == 1:
            u_person = 0.40
            p_desc = "1 isolated potential survivor detected."
        elif person_count <= 3:
            u_person = 0.75
            p_desc = f"Cluster of {person_count} potential survivors detected."
        else:
            u_person = 1.0
            p_desc = f"Group of {person_count} potential survivors detected in close proximity."

        factors.append(
            FactorExplanation(
                factor_name="tracked_person_count",
                raw_value=person_count,
                contribution=f"{u_person * self.assumptions.WEIGHT_URGENCY_PERSONS:.3f}",
                weight_assumption=f"{self.assumptions.WEIGHT_URGENCY_PERSONS:.2f} (Prototype Assumption)",
                impact_description=p_desc,
            )
        )

        # Factor 1.2: Visible Hazard Severity
        hazard_map = {
            HazardSeverity.NONE: (0.0, "No visible environmental hazards detected."),
            HazardSeverity.LOW: (0.20, "Minor hazard observed (e.g. shallow puddle, minor debris)."),
            HazardSeverity.MODERATE: (0.50, "Moderate hazard observed (e.g. localized flooding, blocked lane)."),
            HazardSeverity.HIGH: (0.80, "High hazard observed (e.g. swift water, partial building collapse)."),
            HazardSeverity.CRITICAL: (1.00, "Critical hazard observed (e.g. active fire, raging torrent, severe structural collapse)."),
        }
        u_hazard, h_desc = hazard_map[input_data.hazard_severity]
        factors.append(
            FactorExplanation(
                factor_name="hazard_severity",
                raw_value=input_data.hazard_severity.value,
                contribution=f"{u_hazard * self.assumptions.WEIGHT_URGENCY_HAZARDS:.3f}",
                weight_assumption=f"{self.assumptions.WEIGHT_URGENCY_HAZARDS:.2f} (Prototype Assumption)",
                impact_description=h_desc,
            )
        )

        # Factor 1.3: Accessibility
        if input_data.is_accessible is False:
            u_access = 1.0
            a_desc = "Area egress routes appear blocked, flooded, or impassable."
        elif input_data.is_accessible is True:
            u_access = 0.0
            a_desc = "Area appears accessible via clear surface roads or paths."
        else:
            u_access = 0.50
            a_desc = "Accessibility status could not be verified from aerial angle."
            limitations.append("Ground accessibility is unknown and requires visual road inspection.")

        factors.append(
            FactorExplanation(
                factor_name="accessibility_status",
                raw_value=input_data.is_accessible,
                contribution=f"{u_access * self.assumptions.WEIGHT_URGENCY_INACCESSIBLE:.3f}",
                weight_assumption=f"{self.assumptions.WEIGHT_URGENCY_INACCESSIBLE:.2f} (Prototype Assumption)",
                impact_description=a_desc,
            )
        )

        # Calculate Total Urgency
        raw_urgency = (
            (u_person * self.assumptions.WEIGHT_URGENCY_PERSONS) +
            (u_hazard * self.assumptions.WEIGHT_URGENCY_HAZARDS) +
            (u_access * self.assumptions.WEIGHT_URGENCY_INACCESSIBLE)
        )
        urgency_score = round(min(1.0, max(0.0, raw_urgency)), 3)

        if urgency_score >= 0.75:
            urgency_level = "CRITICAL"
        elif urgency_score >= 0.50:
            urgency_level = "HIGH"
        elif urgency_score >= 0.25:
            urgency_level = "MEDIUM"
        else:
            urgency_level = "LOW"

        # -------------------------------------------------------------
        # 2. UNCERTAINTY EVALUATION: How much verification is needed?
        # -------------------------------------------------------------

        unc_components = []

        # Factor 2.1: Model Confidence
        if input_data.average_detection_confidence is not None:
            c_unc = max(0.0, 1.0 - input_data.average_detection_confidence)
            c_desc = f"Model confidence is {input_data.average_detection_confidence:.2f} (Residual uncertainty: {c_unc:.2f})."
        else:
            c_unc = 0.60
            c_desc = "Detection confidence unrecorded; applying conservative baseline uncertainty."
            limitations.append("Average detection confidence was missing from model payload.")

        unc_components.append(c_unc * 0.35)
        factors.append(
            FactorExplanation(
                factor_name="detection_confidence_uncertainty",
                raw_value=input_data.average_detection_confidence,
                contribution=f"{c_unc * 0.35:.3f}",
                weight_assumption="0.35 (Prototype Assumption)",
                impact_description=c_desc,
            )
        )

        # Factor 2.2: Observation Age (Temporal Drift)
        age = input_data.observation_age_minutes or 0.0
        if age >= self.assumptions.STALE_OBSERVATION_MINUTES * 2:  # > 2 hours
            age_unc = 1.0
            age_desc = f"Observation is severely stale ({age:.1f} minutes old). Ground status likely changed."
            limitations.append(f"Observation is {age:.1f} minutes old; requires fresh aerial recon sweep.")
        elif age >= self.assumptions.STALE_OBSERVATION_MINUTES:     # > 1 hour
            age_unc = 0.65
            age_desc = f"Observation is moderately stale ({age:.1f} minutes old)."
            limitations.append(f"Observation is {age:.1f} minutes old.")
        elif age >= 20.0:
            age_unc = 0.30
            age_desc = f"Observation is {age:.1f} minutes old."
        else:
            age_unc = 0.05
            age_desc = f"Observation is fresh ({age:.1f} minutes old)."

        unc_components.append(age_unc * 0.35)
        factors.append(
            FactorExplanation(
                factor_name="observation_freshness",
                raw_value=f"{age:.1f} min",
                contribution=f"{age_unc * 0.35:.3f}",
                weight_assumption="0.35 (Prototype Assumption)",
                impact_description=age_desc,
            )
        )

        # Factor 2.3: Telemetry Availability
        if input_data.has_missing_telemetry:
            telem_unc = self.assumptions.TELEMETRY_MISSING_PENALTY
            telem_desc = "Missing UAV telemetry; spatial ground coordinates cannot be projected."
            limitations.append("Flight telemetry missing; coordinate estimation unavailable.")
        else:
            telem_unc = 0.0
            telem_desc = "UAV flight telemetry successfully attached."

        unc_components.append(telem_unc)
        factors.append(
            FactorExplanation(
                factor_name="telemetry_integrity",
                raw_value=not input_data.has_missing_telemetry,
                contribution=f"{telem_unc:.3f}",
                weight_assumption=f"{self.assumptions.TELEMETRY_MISSING_PENALTY:.2f} (Penalty if missing)",
                impact_description=telem_desc,
            )
        )

        # Factor 2.4: Image Quality Degradation (smoke, blur, low light)
        deg = input_data.image_quality_degradation
        unc_components.append(deg * 0.20)
        if deg > 0.40:
            limitations.append(f"Image quality degradation detected (index: {deg:.2f}) due to smoke, blur, or darkness.")

        factors.append(
            FactorExplanation(
                factor_name="image_quality_degradation",
                raw_value=deg,
                contribution=f"{deg * 0.20:.3f}",
                weight_assumption="0.20 (Prototype Assumption)",
                impact_description=f"Environmental degradation index is {deg:.2f}.",
            )
        )

        # Calculate Total Uncertainty
        uncertainty_score = round(min(1.0, max(0.05, sum(unc_components))), 3)

        if uncertainty_score >= 0.60:
            uncertainty_level = "HIGH"
        elif uncertainty_score >= 0.35:
            uncertainty_level = "MODERATE"
        else:
            uncertainty_level = "LOW"

        # -------------------------------------------------------------
        # 3. COMPOSITE PRIORITY TIERING
        # -------------------------------------------------------------
        # High urgency or high uncertainty with elevated urgency requires Critical Review
        if urgency_score >= 0.70 or (urgency_score >= 0.45 and uncertainty_score >= 0.60):
            priority_tier = PriorityTier.CRITICAL_REVIEW
        elif urgency_score >= 0.50:
            priority_tier = PriorityTier.HIGH
        elif urgency_score >= 0.25 or uncertainty_score >= 0.50:
            priority_tier = PriorityTier.MEDIUM
        elif urgency_score > 0.05:
            priority_tier = PriorityTier.LOW
        else:
            priority_tier = PriorityTier.INFORMATIONAL

        return PriorityAssessmentResult(
            urgency_score=urgency_score,
            urgency_level=urgency_level,
            uncertainty_score=uncertainty_score,
            uncertainty_level=uncertainty_level,
            composite_priority=priority_tier,
            requires_human_verification=True,  # Safety boundary: Never auto-dispatch
            contributing_factors=factors,
            data_limitations=limitations,
        )
