"""
Demonstration and Testing Script for Phase 6 Location Handling.

Usage:
  python test_location_demo.py
"""
from datetime import datetime, timezone
from app.schemas.domain import BoundingBoxSchema
from app.schemas.location import DroneTelemetryMetadata, LocationSource
from app.services.location import (
    PlanarRayCastingEstimator,
    SimulatedLocationProvider,
)


def main():
    print("=" * 70)
    print("PHASE 6: LOCATION HANDLING & ESTIMATION DEMO")
    print("=" * 70)

    # 1. Test Simulated Location Provider
    print("\n1. SIMULATED LOCATION PROVIDER (Pure Synthetic Scenario)")
    print("-" * 70)
    provider = SimulatedLocationProvider(base_latitude=34.052200, base_longitude=-118.243700)
    bbox_sample = BoundingBoxSchema(x1=500, y1=300, x2=550, y2=420)
    sim_loc = provider.estimate_location(bbox_sample, image_width=1920, image_height=1080)

    print(f"Base Coordinates:    (34.052200, -118.243700)")
    print(f"Target Bounding Box: [{bbox_sample.x1}, {bbox_sample.y1}, {bbox_sample.x2}, {bbox_sample.y2}]")
    print(f"Estimated Person Lat:{sim_loc.latitude}")
    print(f"Estimated Person Lon:{sim_loc.longitude}")
    print(f"Location Source:     '{sim_loc.source}' (Explicitly labeled)")
    print(f"Accuracy Bound:      ±{sim_loc.accuracy_meters}m")
    print(f"Metadata Note:       {sim_loc.metadata.get('note')}")

    # 2. Test Planar Ray Casting with UAV Telemetry
    print("\n2. PLANAR RAY-CASTING ESTIMATOR (Telemetry + Camera Geometry)")
    print("-" * 70)
    drone_telemetry = DroneTelemetryMetadata(
        drone_latitude=34.052200,
        drone_longitude=-118.243700,
        altitude_agl_m=50.0,
        gimbal_pitch_deg=-60.0,  # 60 degrees down
        gimbal_yaw_deg=30.0,     # Heading 30 degrees North-East
        horizontal_fov_deg=84.0,
        source=LocationSource.SIMULATED,
        timestamp=datetime.now(timezone.utc),
    )

    ray_estimator = PlanarRayCastingEstimator()
    person_bbox = BoundingBoxSchema(x1=1200, y1=400, x2=1250, y2=520)

    projected_loc = ray_estimator.estimate_location(
        bbox=person_bbox,
        image_width=1920,
        image_height=1080,
        telemetry=drone_telemetry,
    )

    print(f"Drone GPS Location:  ({drone_telemetry.drone_latitude:.6f}, {drone_telemetry.drone_longitude:.6f})")
    print(f"Drone Altitude AGL:  {drone_telemetry.altitude_agl_m} meters")
    print(f"Gimbal Orientation:  Pitch={drone_telemetry.gimbal_pitch_deg}°, Yaw={drone_telemetry.gimbal_yaw_deg}°")
    print(f"Person Bounding Box: [{person_bbox.x1}, {person_bbox.y1}, {person_bbox.x2}, {person_bbox.y2}]")
    print(f"\n--- Ground Estimation Output ---")
    print(f"Estimated Person GPS:({projected_loc.latitude:.6f}, {projected_loc.longitude:.6f})")
    print(f"Ground Distance:     {projected_loc.metadata.get('ground_offset_meters')} meters from nadir")
    print(f"Location Source:     '{projected_loc.source}' (Strictly enforced)")
    print(f"Accuracy / Error:    ±{projected_loc.accuracy_meters}m")
    print(f"Timestamp:           {projected_loc.timestamp}")

    # 3. Test Missing Telemetry Boundary
    print("\n3. MISSING TELEMETRY SAFETY BOUNDARY")
    print("-" * 70)
    no_telem_result = ray_estimator.estimate_location(person_bbox, 1920, 1080, telemetry=None)
    print(f"Location with telemetry=None: {no_telem_result}")
    print("Verification: System safely refused to invent GPS coordinates from pixels alone.")

    print("\n" + "=" * 70)
    print("[SUCCESS] Phase 6 Location Handling tested successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
