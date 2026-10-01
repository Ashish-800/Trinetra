"""
Location Handling Service.
Provides decoupled location estimation interfaces, enforcing:
- Mandatory source tagging (simulated vs real)
- Uncertainty/accuracy error radii
- Never inferring real GPS from pixels alone
- Never equating drone coordinates directly to person ground coordinates
"""
import math
from abc import ABC, abstractmethod
from typing import Optional
from app.schemas.domain import BoundingBoxSchema
from app.schemas.location import DroneTelemetryMetadata, LocationRecord, LocationSource


class LocationError(Exception):
    """Base exception for location service errors."""
    pass


class InvalidCoordinateError(LocationError):
    """Raised when latitude or longitude coordinates fall outside valid WGS-84 ranges."""
    pass


class MissingTelemetryError(LocationError):
    """Raised when geolocation is attempted without necessary flight telemetry."""
    pass


class BaseLocationEstimator(ABC):
    """Abstract interface decoupling location estimation algorithms."""

    @abstractmethod
    def estimate_location(
        self,
        bbox: BoundingBoxSchema,
        image_width: int,
        image_height: int,
        telemetry: Optional[DroneTelemetryMetadata] = None,
    ) -> Optional[LocationRecord]:
        """
        Estimates the approximate ground coordinate for a detected object.
        Returns None if telemetry is missing and coordinates cannot be computed.
        """
        pass


class SimulatedLocationProvider(BaseLocationEstimator):
    """
    Supplies explicitly simulated coordinates for synthetic scenarios and tests.
    Every produced location record is strictly tagged with source='simulated'.
    """

    def __init__(
        self,
        base_latitude: float = 34.0522,
        base_longitude: float = -118.2437,
        default_accuracy_m: float = 15.0,
    ):
        self._validate_coords(base_latitude, base_longitude)
        self.base_latitude = base_latitude
        self.base_longitude = base_longitude
        self.default_accuracy_m = default_accuracy_m

    @staticmethod
    def _validate_coords(lat: float, lon: float):
        if not (-90.0 <= lat <= 90.0):
            raise InvalidCoordinateError(f"Latitude {lat} is outside valid range [-90.0, 90.0].")
        if not (-180.0 <= lon <= 180.0):
            raise InvalidCoordinateError(f"Longitude {lon} is outside valid range [-180.0, 180.0].")

    def estimate_location(
        self,
        bbox: BoundingBoxSchema,
        image_width: int,
        image_height: int,
        telemetry: Optional[DroneTelemetryMetadata] = None,
    ) -> LocationRecord:
        """
        Calculates a deterministic simulated coordinate based on bounding box centroid.
        Explicitly marked with source='simulated'.
        """
        # Centroid normalized displacement from image center [-0.5, 0.5]
        cx = (bbox.x1 + bbox.x2) / (2.0 * max(1, image_width)) - 0.5
        cy = (bbox.y1 + bbox.y2) / (2.0 * max(1, image_height)) - 0.5

        # Offset by small simulated distance (~10-50 meters)
        delta_lat = cy * 0.0005
        delta_lon = cx * 0.0005

        sim_lat = round(self.base_latitude + delta_lat, 6)
        sim_lon = round(self.base_longitude + delta_lon, 6)

        return LocationRecord(
            latitude=sim_lat,
            longitude=sim_lon,
            source=LocationSource.SIMULATED,
            accuracy_meters=self.default_accuracy_m,
            metadata={
                "estimation_method": "synthetic_grid_offset",
                "is_synthetic": True,
                "note": "Explicitly simulated coordinate for prototype demonstration.",
            },
        )


class PlanarRayCastingEstimator(BaseLocationEstimator):
    """
    Projects camera rays to a planar flat ground surface using UAV telemetry.
    
    Safety Rules Enforced:
    1. Does NOT infer real-world GPS from pixels alone; requires telemetry.
    2. Does NOT copy drone GPS as person ground coordinate; computes ground ray offset.
    3. Outputs explicit error radii (accuracy_meters) reflecting altitude & oblique angles.
    4. Explicitly tags resulting coordinates as source='simulated' (since flat terrain is assumed).
    """

    METERS_PER_DEG_LAT = 111320.0  # WGS-84 standard approximation

    def estimate_location(
        self,
        bbox: BoundingBoxSchema,
        image_width: int,
        image_height: int,
        telemetry: Optional[DroneTelemetryMetadata] = None,
    ) -> Optional[LocationRecord]:
        """
        Projects 2D image coordinates to approximate ground coordinates using telemetry.
        Returns None if telemetry is not supplied.
        """
        if telemetry is None:
            return None

        # Normalized centroid coordinates relative to center ([-1, 1])
        bx = (bbox.x1 + bbox.x2) / 2.0
        by = (bbox.y1 + bbox.y2) / 2.0

        u = (bx - image_width / 2.0) / (image_width / 2.0)
        v = (by - image_height / 2.0) / (image_height / 2.0)

        # Field of view angles
        half_hfov_rad = math.radians(telemetry.horizontal_fov_deg / 2.0)
        # Assuming standard 4:3 or 16:9 aspect ratio for vertical FOV
        aspect_ratio = max(0.1, image_width / max(1, image_height))
        half_vfov_rad = half_hfov_rad / aspect_ratio

        # Ray angles relative to camera optical axis
        ray_yaw_rel = u * half_hfov_rad
        ray_pitch_rel = -v * half_vfov_rad

        # Total angles relative to ground
        effective_pitch = math.radians(telemetry.gimbal_pitch_deg) + ray_pitch_rel
        effective_heading = math.radians(telemetry.gimbal_yaw_deg) + ray_yaw_rel

        # Ground intersection (flat earth assumption)
        altitude = telemetry.altitude_agl_m
        tan_pitch = math.tan(abs(effective_pitch))
        if tan_pitch < 0.05:  # Horizon or skyward ray, no ground intersection
            tan_pitch = 0.05

        ground_distance_m = altitude / tan_pitch

        # Offsets in meters North and East of drone
        offset_north = ground_distance_m * math.cos(effective_heading)
        offset_east = ground_distance_m * math.sin(effective_heading)

        # Convert meter offsets to delta degrees
        d_lat = offset_north / self.METERS_PER_DEG_LAT
        cos_lat = math.cos(math.radians(telemetry.drone_latitude))
        meters_per_deg_lon = self.METERS_PER_DEG_LAT * max(0.1, abs(cos_lat))
        d_lon = offset_east / meters_per_deg_lon

        estimated_lat = round(telemetry.drone_latitude + d_lat, 6)
        estimated_lon = round(telemetry.drone_longitude + d_lon, 6)

        # Accuracy estimation: error radius increases with altitude and ground distance
        # Standard consumer GPS drift (~5m) + projection uncertainty (~10% of distance)
        estimated_accuracy_m = round(5.0 + (ground_distance_m * 0.10) + (altitude * 0.05), 1)

        return LocationRecord(
            latitude=estimated_lat,
            longitude=estimated_lon,
            source=LocationSource.SIMULATED,  # Planar projection without DEM is simulated approximation
            accuracy_meters=estimated_accuracy_m,
            timestamp=telemetry.timestamp,
            metadata={
                "estimation_method": "planar_ray_projection",
                "drone_altitude_m": altitude,
                "gimbal_pitch_deg": telemetry.gimbal_pitch_deg,
                "ground_offset_meters": round(ground_distance_m, 2),
                "drone_latitude": telemetry.drone_latitude,
                "drone_longitude": telemetry.drone_longitude,
                "note": "Approximate ground coordinate derived from planar ray projection. Not calibrated by DEM.",
            },
        )
