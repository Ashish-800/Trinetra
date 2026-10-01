"""
Demonstration and Manual Testing Script for Phase 8 Resource Recommendation.

Usage:
  python test_recommender_demo.py
"""
from app.schemas.location import LocationRecord, LocationSource
from app.schemas.resource_engine import IncidentRequirement
from app.services.recommender import (
    ResourceRecommendationService,
    get_default_simulated_inventory,
)


def print_outcome(title: str, outcome):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {title}")
    print("=" * 70)
    print(f"Status:   {outcome.status}")
    print(f"Summary:  {outcome.incident_summary}")

    if outcome.recommendations:
        print("\n--- Advisory Recommendations ---")
        for idx, rec in enumerate(outcome.recommendations, 1):
            print(f"  [{idx}] {rec.resource_name} ({rec.resource_id})")
            print(f"      Type:             {rec.resource_type}")
            print(f"      Suitability:      {rec.suitability_score:.2f}")
            print(f"      Matched Caps:     {', '.join(rec.matched_capabilities)}")
            print(f"      Distance Info:    {rec.distance_note}")
            print(f"      Rationale:        {rec.rationale}")
            print(f"      Human Auth Gate:  {rec.requires_human_authorization}")

    if outcome.unsuitable_resources_reasoning:
        print("\n--- Unsuitable Resources & Rejection Reasons ---")
        for res_id, reason in outcome.unsuitable_resources_reasoning.items():
            print(f"  x {res_id:15}: {reason}")

    print(f"\nDisclaimer: {outcome.advisory_disclaimer}")


def main():
    service = ResourceRecommendationService()
    sim_loc = LocationRecord(latitude=34.055, longitude=-118.245, source=LocationSource.SIMULATED)

    # 1. Available resource match with valid location
    print_outcome(
        "1. Active Flood Zone (Capable Resource Available with Distance)",
        service.recommend(
            IncidentRequirement(
                incident_id="INC-01",
                disaster_type="FLOOD",
                required_capabilities=["flood_rescue"],
                needed_capacity=3,
                target_location=sim_loc,
            )
        ),
    )

    # 2. Missing location scenario (evaluates without fabricating distance)
    print_outcome(
        "2. Earthquake Search (Missing Target Location - Distance Uncalculated)",
        service.recommend(
            IncidentRequirement(
                incident_id="INC-02",
                disaster_type="EARTHQUAKE",
                required_capabilities=["debris_clearing", "first_aid"],
                needed_capacity=2,
                target_location=None,  # Missing location!
            )
        ),
    )

    # 3. Constraint violation (night conditions block daylight-only units)
    print_outcome(
        "3. Night Reconnaissance (Constraint Violation Excludes Incompatible Units)",
        service.recommend(
            IncidentRequirement(
                incident_id="INC-03",
                disaster_type="NIGHT_SEARCH",
                required_capabilities=["debris_clearing"],
                needed_capacity=1,
                environmental_conditions=["night", "impassable_roads"],
            )
        ),
    )

    # 4. No suitable resource available
    print_outcome(
        "4. High-Altitude Alpine Rescue (No Suitable Inventory Resource)",
        service.recommend(
            IncidentRequirement(
                incident_id="INC-04",
                disaster_type="ALPINE_AVALANCHE",
                required_capabilities=["avalanche_crevasse_extraction"],
                needed_capacity=1,
            )
        ),
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] All Phase 8 Resource Recommendation scenarios evaluated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
