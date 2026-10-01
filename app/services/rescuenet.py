"""
RescueNet Dataset Processing and Decision-Support Service.
Analyzes ultra-high-resolution aerial drone disaster segmentation masks,
translating pixel-level building damage, flood water, and debris into
standardized priority inputs and resource capability recommendations.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zlib
import struct

from app.core.config import settings
from app.schemas.priority import AssessmentInput, HazardSeverity
from app.schemas.rescuenet import (
    CLASS_DISPLAY_NAMES,
    RescueNetClass,
    RescueNetHazardThresholds,
    RescueNetSceneAnalysis,
)
from app.schemas.resource_engine import IncidentRequirement


class RescueNetDatasetError(Exception):
    """Base exception for RescueNet dataset processing."""
    pass


class RescueNetService:
    """
    Service for parsing and integrating the RescueNet UAV post-disaster dataset.
    Extracts macro-environmental context from aerial semantic segmentation masks.
    NOTE: RescueNet provides macro-environmental disaster context (water, debris, collapse).
    It does NOT detect people or determine victim survival status.
    """

    DEFAULT_RESCUENET_DIR = settings.BASE_DIR / "RescueNet"

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        thresholds: Optional[RescueNetHazardThresholds] = None,
    ):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else self.DEFAULT_RESCUENET_DIR
        self.thresholds = thresholds or RescueNetHazardThresholds()

    def is_available(self) -> bool:
        """Returns True if the RescueNet directory exists and contains expected subfolders."""
        return (self.dataset_dir / "val").exists() or (self.dataset_dir / "train").exists()

    def list_samples(self, split: str = "val", max_count: Optional[int] = None) -> List[Tuple[Path, Path]]:
        """
        Discovers matched (image_path, mask_path) pairs for a dataset split ('train', 'val', 'test').
        """
        split_dir = self.dataset_dir / split
        img_dir = split_dir / f"{split}-org-img"
        lab_dir = split_dir / f"{split}-label-img"

        if not img_dir.exists() or not lab_dir.exists():
            return []

        samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            mask_path = lab_dir / f"{stem}_lab.png"
            if mask_path.exists():
                samples.append((img_path, mask_path))
                if max_count and len(samples) >= max_count:
                    break
        return samples

    @staticmethod
    def _read_png_dimensions_and_pixels_fast(mask_path: Path) -> Tuple[int, int, Dict[int, int]]:
        """
        Reads PNG dimensions and counts pixel values directly without external heavy dependencies.
        Uses OpenCV if installed; falls back to standard library zlib decompression if offline.
        """
        # 1. Try OpenCV if available
        try:
            import cv2
            import numpy as np
            mat = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mat is not None:
                h, w = mat.shape
                uniques, counts = np.unique(mat, return_counts=True)
                pixel_counts = {int(u): int(c) for u, c in zip(uniques, counts)}
                return w, h, pixel_counts
        except Exception:
            pass

        # 2. Try PIL if available
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(mask_path)
            w, h = img.size
            arr = np.array(img)
            uniques, counts = np.unique(arr, return_counts=True)
            pixel_counts = {int(u): int(c) for u, c in zip(uniques, counts)}
            return w, h, pixel_counts
        except Exception:
            pass

        # 3. Standard Library Fallback (parse PNG chunks)
        with open(mask_path, "rb") as f:
            header = f.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                raise RescueNetDatasetError(f"Invalid PNG signature in {mask_path}")

            idat_chunks = []
            width = 0
            height = 0

            while True:
                chunk_len_bytes = f.read(4)
                if not chunk_len_bytes:
                    break
                chunk_len = struct.unpack(">I", chunk_len_bytes)[0]
                chunk_type = f.read(4)
                chunk_data = f.read(chunk_len)
                f.read(4)  # CRC

                if chunk_type == b"IHDR":
                    width, height = struct.unpack(">II", chunk_data[:8])
                elif chunk_type == b"IDAT":
                    idat_chunks.append(chunk_data)
                elif chunk_type == b"IEND":
                    break

            # Decompress raw scanlines (1 byte per pixel grayscale)
            try:
                raw_bytes = zlib.decompress(b"".join(idat_chunks))
                # Count byte frequencies directly
                pixel_counts = {}
                for b in raw_bytes:
                    if b <= 11:  # valid RescueNet class index
                        pixel_counts[b] = pixel_counts.get(b, 0) + 1
                return width, height, pixel_counts
            except Exception:
                # Fallback header info
                return width, height, {0: width * height}

    def _build_scene_analysis(
        self,
        image_id: str,
        total_pixels: int,
        pixel_counts: Dict[int, int],
        thresholds: Optional[RescueNetHazardThresholds] = None,
    ) -> RescueNetSceneAnalysis:
        """Helper to build RescueNetSceneAnalysis from aggregated pixel counts."""
        # Compute percentage for each class
        distribution: Dict[str, float] = {}
        for cls_enum in RescueNetClass:
            count = pixel_counts.get(int(cls_enum), 0)
            pct = round((count / total_pixels) * 100.0, 3)
            distribution[CLASS_DISPLAY_NAMES[cls_enum]] = pct

        water_pct = distribution.get("Water", 0.0)
        debris_pct = distribution.get("Debris", 0.0)
        destroyed_pct = distribution.get("Building Total Destruction", 0.0)
        major_pct = distribution.get("Building Major Damage", 0.0)
        minor_pct = distribution.get("Building Minor Damage", 0.0)
        intact_pct = distribution.get("Building No Damage", 0.0)
        road_pct = distribution.get("Road", 0.0)
        vehicle_pct = distribution.get("Vehicle", 0.0)
        tree_pct = distribution.get("Tree", 0.0)
        pool_pct = distribution.get("Pool", 0.0)
        sand_pct = distribution.get("Sand", 0.0)

        # 1. Infer Hazard Severity via documented, configurable thresholds
        active_thresh = thresholds or self.thresholds
        contributing_factors = []
        if (
            destroyed_pct >= active_thresh.critical_destroyed_building_pct
            or water_pct >= active_thresh.critical_water_pct
        ):
            hazard = HazardSeverity.CRITICAL
            if destroyed_pct >= active_thresh.critical_destroyed_building_pct:
                contributing_factors.append(f"Catastrophic Building Destruction ({destroyed_pct}% footprint)")
            if water_pct >= active_thresh.critical_water_pct:
                contributing_factors.append(f"Severe Flood Inundation ({water_pct}% surface coverage)")
        elif (
            major_pct >= active_thresh.high_major_building_pct
            or water_pct >= active_thresh.high_water_pct
            or debris_pct >= active_thresh.high_debris_pct
        ):
            hazard = HazardSeverity.HIGH
            if major_pct >= active_thresh.high_major_building_pct:
                contributing_factors.append(f"Major Structural Failure ({major_pct}%)")
            if water_pct >= active_thresh.high_water_pct:
                contributing_factors.append(f"Flood Waters Present ({water_pct}%)")
            if debris_pct >= active_thresh.high_debris_pct:
                contributing_factors.append(f"Heavy Debris Field ({debris_pct}%)")
        elif (
            debris_pct >= active_thresh.moderate_debris_pct
            or water_pct >= active_thresh.moderate_water_pct
            or minor_pct >= active_thresh.moderate_minor_building_pct
        ):
            hazard = HazardSeverity.MODERATE
            contributing_factors.append("Moderate debris and partial structural damage")
        elif (
            minor_pct >= active_thresh.low_minor_building_pct
            or debris_pct >= active_thresh.low_debris_pct
        ):
            hazard = HazardSeverity.LOW
            contributing_factors.append("Minor non-structural debris or superficial damage")
        else:
            hazard = HazardSeverity.NONE
            contributing_factors.append("No significant structural or flood hazards detected")

        # 2. Infer Terrain Accessibility
        if debris_pct >= 15.0 or water_pct >= 25.0:
            accessibility = "isolated"
            contributing_factors.append("Road networks severely obstructed or submerged")
        elif debris_pct >= 6.0 or water_pct >= 8.0 or road_pct < 1.0:
            accessibility = "difficult"
            contributing_factors.append("Roadways partially blocked by debris or standing water")
        else:
            accessibility = "accessible"

        # 3. Infer Required Capabilities
        capabilities = []
        if water_pct >= 4.0:
            capabilities.append("WATER_RESCUE")
        if destroyed_pct >= 0.3:
            capabilities.extend(["HEAVY_DEBRIS_CLEARING", "CANINE_SEARCH", "STRUCTURAL_SHORING"])
        elif debris_pct >= 6.0:
            capabilities.append("HEAVY_DEBRIS_CLEARING")
        if vehicle_pct > 0.05:
            capabilities.append("GROUND_TRANSPORT")
        if hazard in (HazardSeverity.CRITICAL, HazardSeverity.HIGH):
            capabilities.append("TRIAGE_MEDICAL")

        return RescueNetSceneAnalysis(
            image_id=image_id,
            total_pixels=total_pixels,
            class_distribution_pct=distribution,
            water_coverage_pct=water_pct,
            debris_coverage_pct=debris_pct,
            destroyed_building_pct=destroyed_pct,
            major_damage_building_pct=major_pct,
            minor_damage_building_pct=minor_pct,
            intact_building_pct=intact_pct,
            road_coverage_pct=road_pct,
            tree_coverage_pct=tree_pct,
            pool_coverage_pct=pool_pct,
            sand_coverage_pct=sand_pct,
            vehicle_detected=vehicle_pct > 0.02,
            inferred_hazard_severity=hazard,
            inferred_accessibility=accessibility,
            inferred_required_capabilities=list(set(capabilities)),
            contributing_hazard_factors=contributing_factors,
            thresholds_used=active_thresh,
        )

    def analyze_scene(
        self,
        mask_path: Union[str, Path],
        thresholds: Optional[RescueNetHazardThresholds] = None,
    ) -> RescueNetSceneAnalysis:
        """
        Analyzes a RescueNet semantic segmentation mask file and maps pixel distributions
        into environmental disaster parameters and required capabilities using
        explicit, transparent decision thresholds.
        """
        path = Path(mask_path)
        if not path.exists():
            raise RescueNetDatasetError(f"RescueNet label mask not found at: {path}")

        image_id = path.stem.replace("_lab", "")
        width, height, pixel_counts = self._read_png_dimensions_and_pixels_fast(path)
        total_pixels = max(1, width * height)
        return self._build_scene_analysis(
            image_id=image_id,
            total_pixels=total_pixels,
            pixel_counts=pixel_counts,
            thresholds=thresholds,
        )

    def analyze_mask_array(
        self,
        mask_array: Any,
        image_id: str = "pred_mask",
        thresholds: Optional[RescueNetHazardThresholds] = None,
    ) -> RescueNetSceneAnalysis:
        """
        Analyzes an in-memory 2D segmentation map (numpy array or PyTorch tensor)
        with integer class labels in 0..11 and computes RescueNetSceneAnalysis.
        """
        import numpy as np
        if hasattr(mask_array, "cpu"):
            mask_arr = mask_array.detach().cpu().numpy()
        else:
            mask_arr = np.asarray(mask_array)

        total_pixels = int(mask_arr.size)
        uniques, counts = np.unique(mask_arr, return_counts=True)
        pixel_counts = {int(u): int(c) for u, c in zip(uniques, counts)}

        return self._build_scene_analysis(
            image_id=image_id,
            total_pixels=max(1, total_pixels),
            pixel_counts=pixel_counts,
            thresholds=thresholds,
        )

    def analyze_sample(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        thresholds: Optional[RescueNetHazardThresholds] = None,
    ) -> RescueNetSceneAnalysis:
        """
        Loads and analyzes a RescueNet image and its corresponding segmentation mask.
        If mask_path is omitted, attempts to discover the matching '_lab.png' in the dataset.
        """
        img_p = Path(image_path)
        if not img_p.exists():
            raise RescueNetDatasetError(f"RescueNet image not found at: {img_p}")

        if mask_path is not None:
            resolved_mask = Path(mask_path)
        else:
            parent_name = img_p.parent.name
            if "org-img" in parent_name:
                label_dir = img_p.parent.parent / parent_name.replace("org-img", "label-img")
                resolved_mask = label_dir / f"{img_p.stem}_lab.png"
            else:
                resolved_mask = img_p.parent / f"{img_p.stem}_lab.png"

        if not resolved_mask.exists():
            raise RescueNetDatasetError(
                f"Could not locate corresponding RescueNet mask for '{img_p.name}'. Checked '{resolved_mask}'"
            )

        return self.analyze_scene(mask_path=resolved_mask, thresholds=thresholds)

    def to_assessment_input(
        self,
        scene: RescueNetSceneAnalysis,
        distinct_person_tracks: int = 1,
        mean_confidence: float = 0.88,
        observation_age_seconds: float = 30.0,
    ) -> AssessmentInput:
        """
        Converts a RescueNet scene analysis into structured inputs for PriorityVerificationEngine.
        """
        return AssessmentInput(
            distinct_person_track_count=distinct_person_tracks,
            hazard_severity=scene.inferred_hazard_severity,
            terrain_accessibility=scene.inferred_accessibility or "accessible",
            mean_detection_confidence=mean_confidence,
            observation_age_seconds=observation_age_seconds,
        )

    def to_incident_requirement(
        self,
        scene: RescueNetSceneAnalysis,
        incident_lat: float = 30.1588,
        incident_lon: float = -85.6602,  # Hurricane Michael epicenter (Panama City, FL)
        required_capacity: int = 2,
    ) -> IncidentRequirement:
        """
        Converts a RescueNet scene analysis into structured IncidentRequirement for ResourceRecommendationService.
        """
        from app.schemas.resource_engine import ResourceCapability
        caps = []
        for c in scene.inferred_required_capabilities:
            try:
                caps.append(ResourceCapability(c))
            except ValueError:
                pass
        
        primary_cap = caps[0] if caps else ResourceCapability.GROUND_TRANSPORT
        return IncidentRequirement(
            required_capability=primary_cap,
            required_capacity=required_capacity,
            incident_lat=incident_lat,
            incident_lon=incident_lon,
        )
