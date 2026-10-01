"""
Integration Tests for Phase 10: FastAPI REST API.
Tests all endpoints using TestClient with an isolated in-memory database and temporary uploads.
"""
import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_detector
from app.core.config import settings
from app.main import app
from app.schemas.domain import BoundingBoxSchema
from app.services.detector import DetectedObject, MockDetector


@pytest.fixture
def client(db_session: Session, tmp_path: Path):
    """Configures TestClient with overridden database session and isolated upload directory."""
    # Point upload dir to temp path
    original_upload_dir = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = tmp_path / "test_uploads"
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Use predictable mock detector
    mock_detector = MockDetector(
        canned_detections=[
            DetectedObject.create_safe(
                class_name="person",
                confidence=0.91,
                bbox=BoundingBoxSchema(x1=100.0, y1=100.0, x2=150.0, y2=200.0),
            )
        ]
    )

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_detector] = lambda: mock_detector

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    settings.UPLOAD_DIR = original_upload_dir


def test_health_endpoints(client: TestClient):
    """Tests both root and v1 health diagnostic endpoints."""
    # Root /health
    res_root = client.get("/health")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "healthy"
    assert "version" in data_root
    assert "safety_constitution" in data_root

    # /api/v1/health
    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "healthy"


def test_incident_lifecycle(client: TestClient):
    """Tests incident creation, validation, listing, and single retrieval."""
    # 1. Create valid incident
    payload = {
        "title": "Flood Sector Delta-4",
        "disaster_type": "FLOOD",
        "location_name": "Riverside County, Grid 12",
        "notes": "Severe flash flood reported near main bridge.",
    }
    create_res = client.post("/api/v1/incidents/", json=payload)
    assert create_res.status_code == 201
    inc_data = create_res.json()
    inc_id = inc_data["id"]
    assert inc_data["title"] == payload["title"]
    assert inc_data["is_active"] is True

    # 2. Validation error: title too short (< 3 chars)
    bad_res = client.post("/api/v1/incidents/", json={"title": "X", "location_name": "Test"})
    assert bad_res.status_code == 422

    # 3. List incidents
    list_res = client.get("/api/v1/incidents/")
    assert list_res.status_code == 200
    incidents = list_res.json()
    assert len(incidents) >= 1
    assert any(i["id"] == inc_id for i in incidents)

    # 4. Get specific incident
    get_res = client.get(f"/api/v1/incidents/{inc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == inc_id

    # 5. Non-existent incident
    missing_res = client.get("/api/v1/incidents/99999")
    assert missing_res.status_code == 404


def test_media_upload_and_validation(client: TestClient):
    """Tests media file upload, format validation, and retrieval."""
    # Create incident first
    inc_res = client.post(
        "/api/v1/incidents/",
        json={"title": "Alpine Mudslide Area", "location_name": "North Valley"},
    )
    inc_id = inc_res.json()["id"]

    # 1. Create a synthetic JPEG in memory
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded_img = cv2.imencode(".jpg", img)
    image_bytes = io.BytesIO(encoded_img.tobytes())

    # 2. Upload valid image
    upload_res = client.post(
        "/api/v1/media/upload",
        data={"incident_id": inc_id},
        files={"file": ("recon_flight.jpg", image_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 201
    media_data = upload_res.json()
    asset_id = media_data["id"]
    assert media_data["incident_id"] == inc_id
    assert media_data["media_type"] == "IMAGE"
    assert media_data["width_px"] == 100
    assert media_data["height_px"] == 100

    # 3. Retrieve media asset metadata
    get_res = client.get(f"/api/v1/media/{asset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == asset_id

    # 4. Reject upload for non-existent incident
    bad_inc_res = client.post(
        "/api/v1/media/upload",
        data={"incident_id": 99999},
        files={"file": ("recon.jpg", io.BytesIO(b"data"), "image/jpeg")},
    )
    assert bad_inc_res.status_code == 404

    # 5. Reject unsupported file extension
    bad_ext_res = client.post(
        "/api/v1/media/upload",
        data={"incident_id": inc_id},
        files={"file": ("malicious.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert bad_ext_res.status_code == 400
    assert "Unsupported file format" in bad_ext_res.json()["detail"]


def test_start_analysis_and_query_results(client: TestClient):
    """
    Tests complete workflow execution via API:
    Upload -> Start Analysis -> Inspect Detections, Tracks, Priority, and Recommendations.
    """
    # 1. Create incident and seed a simulated resource
    inc_res = client.post(
        "/api/v1/incidents/",
        json={"title": "Coastal Storm Surge", "location_name": "Harbor Zone 3"},
    )
    inc_id = inc_res.json()["id"]

    client.post(
        "/api/v1/resources/",
        json={
            "name": "Rapid Water Rescue Team Alpha",
            "resource_type": "WATER_RESCUE",
            "total_capacity": 6,
            "available_capacity": 6,
            "simulated_lat": 34.053,
            "simulated_lon": -118.245,
            "is_available": True,
        },
    )

    # 2. Upload synthetic aerial image
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    upload_res = client.post(
        "/api/v1/media/upload",
        data={"incident_id": inc_id},
        files={"file": ("coastal_recon.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    asset_id = upload_res.json()["id"]

    # 3. Start analysis
    analysis_res = client.post(
        "/api/v1/analysis/start",
        json={"media_asset_id": asset_id, "sampling_interval_seconds": 1.0},
    )
    assert analysis_res.status_code == 200
    summary = analysis_res.json()
    assert summary["incident_id"] == inc_id
    assert summary["media_asset_id"] == asset_id
    assert summary["total_detections_found"] == 1
    assert summary["unique_person_tracks"] == 1
    assert "priority_assessment" in summary
    assert "recommendation_outcome" in summary

    # 4. Retrieve detections for the incident
    det_res = client.get(f"/api/v1/incidents/{inc_id}/detections")
    assert det_res.status_code == 200
    detections = det_res.json()
    assert len(detections) == 1
    det_id = detections[0]["id"]
    assert detections[0]["class_name"] == "person"
    assert detections[0]["location_source"] == "simulated"

    # Single detection retrieval
    single_det = client.get(f"/api/v1/detections/{det_id}")
    assert single_det.status_code == 200
    assert single_det.json()["id"] == det_id

    # 5. Retrieve tracks for the incident
    track_res = client.get(f"/api/v1/incidents/{inc_id}/tracks")
    assert track_res.status_code == 200
    tracks = track_res.json()
    assert len(tracks) == 1

    # 6. Retrieve priority assessments
    prio_res = client.get(f"/api/v1/incidents/{inc_id}/priority-assessments")
    assert prio_res.status_code == 200
    assessments = prio_res.json()
    assert len(assessments) >= 1
    assert "urgency_score" in assessments[0]
    assert "uncertainty_score" in assessments[0]

    # 7. Retrieve resource recommendations
    rec_res = client.get(f"/api/v1/incidents/{inc_id}/recommendations")
    assert rec_res.status_code == 200
    recs = rec_res.json()
    assert len(recs) >= 1
    # Check Project Safety Rule: must require human authorization
    assert recs[0]["requires_human_authorization"] is True


def test_resource_inventory_management(client: TestClient):
    """Tests creating, querying, and updating simulated resources."""
    # 1. Create simulated resource
    payload = {
        "name": "Heavy Aerial Evac Drone 01",
        "resource_type": "AERIAL_EVAC",
        "total_capacity": 2,
        "available_capacity": 2,
        "simulated_lat": 34.050,
        "simulated_lon": -118.250,
        "is_available": True,
    }
    create_res = client.post("/api/v1/resources/", json=payload)
    assert create_res.status_code == 201
    res_data = create_res.json()
    res_id = res_data["id"]
    assert res_data["name"] == payload["name"]

    # 2. List resources with type filter
    list_res = client.get("/api/v1/resources/?resource_type=AERIAL_EVAC")
    assert list_res.status_code == 200
    items = list_res.json()
    assert len(items) >= 1
    assert items[0]["id"] == res_id

    # 3. Patch resource (dispatch 1 seat)
    patch_res = client.patch(f"/api/v1/resources/{res_id}", json={"available_capacity": 1})
    assert patch_res.status_code == 200
    assert patch_res.json()["available_capacity"] == 1

    # 4. Reject invalid capacity update (available > total)
    bad_patch = client.patch(f"/api/v1/resources/{res_id}", json={"available_capacity": 10})
    assert bad_patch.status_code == 400

    # 5. Get missing resource
    missing = client.get("/api/v1/resources/88888")
    assert missing.status_code == 404


def test_analysis_endpoint_with_segmentation(client: TestClient):
    """
    Tests that POST /api/v1/analysis/start with enable_segmentation=True
    runs the combined workflow and populates scene_context alongside detections.
    """
    # 1. Create incident
    inc_res = client.post(
        "/api/v1/incidents/",
        json={"title": "Multimodal Sector Recon", "location_name": "Grid B-2"},
    )
    inc_id = inc_res.json()["id"]

    # 2. Upload synthetic aerial image
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    upload_res = client.post(
        "/api/v1/media/upload",
        data={"incident_id": inc_id},
        files={"file": ("recon_seg.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    asset_id = upload_res.json()["id"]

    # 3. Start analysis with enable_segmentation=True
    analysis_res = client.post(
        "/api/v1/analysis/start",
        json={"media_asset_id": asset_id, "enable_segmentation": True},
    )
    assert analysis_res.status_code == 200
    summary = analysis_res.json()
    assert summary["incident_id"] == inc_id
    assert summary["media_asset_id"] == asset_id
    assert summary["total_detections_found"] == 1
    assert "priority_assessment" in summary
    assert "recommendation_outcome" in summary
    # Verify scene context is populated
    assert summary.get("scene_context") is not None
    assert "class_distribution_pct" in summary["scene_context"]


def test_multimodal_single_request_endpoint(client: TestClient):
    """
    Tests the single-request POST /api/v1/analysis/multimodal endpoint:
    accepts an uploaded media file, executes YOLO + RescueNet segmentation + workflow,
    and returns the combined structured result.
    """
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)

    res = client.post(
        "/api/v1/analysis/multimodal",
        data={"enable_segmentation": True},
        files={"file": ("aerial_multimodal.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["incident_id"] > 0
    assert data["media_asset_id"] > 0
    assert data["total_detections_found"] == 1
    assert data["unique_person_tracks"] == 1

    # 1. Incident information
    assert data.get("incident") is not None
    assert data["incident"]["id"] == data["incident_id"]
    assert "title" in data["incident"]
    assert "disaster_type" in data["incident"]

    # 2. Media information
    assert data.get("media") is not None
    assert data["media"]["id"] == data["media_asset_id"]
    assert data["media"]["media_type"] == "IMAGE"
    assert "file_path" in data["media"]
    assert data["media"]["file_size_bytes"] > 0

    # 3. Potential-person detections list & safe terminology
    assert len(data.get("detections", [])) == 1
    det = data["detections"][0]
    assert det["label"] in ("potential person", "person detected")
    assert "survivor" not in det["label"].lower() or "potential" in det["label"].lower()
    assert det["location_source"] == "simulated"
    assert "bbox" in det
    assert det["bbox"]["x1"] <= det["bbox"]["x2"]
    assert det["confidence"] > 0.0
    assert det["review_status"] == "PENDING_REVIEW"

    # 4. Environmental segmentation summary
    assert data["scene_context"] is not None
    ctx = data["scene_context"]
    assert "water_coverage_pct" in ctx
    assert "debris_coverage_pct" in ctx
    assert "road_coverage_pct" in ctx
    assert "inferred_hazard_severity" in ctx
    assert "inferred_accessibility" in ctx
    assert "inferred_required_capabilities" in ctx

    # 5. Priority and uncertainty metrics
    assert "priority_assessment" in data
    prio = data["priority_assessment"]
    assert "urgency_score" in prio
    assert "urgency_level" in prio
    assert "uncertainty_score" in prio
    assert "uncertainty_level" in prio
    assert "composite_priority" in prio
    assert prio["requires_human_verification"] is True

    # 6. Advisory resource recommendations
    assert "recommendation_outcome" in data
    outcome = data["recommendation_outcome"]
    assert outcome["status"] in ("RECOMMENDATION_AVAILABLE", "NO_SUITABLE_RESOURCE_AVAILABLE")
    for rec in outcome["recommendations"]:
        assert rec["requires_human_authorization"] is True
        assert rec["status"] == "PENDING_REVIEW"

    # 7. Device information
    assert data.get("device_info") is not None
    dev = data["device_info"]
    assert "cuda_available" in dev
    assert "inference_device" in dev
    assert "detector_model" in dev
    assert "segmentation_model" in dev

    # 8. Safety disclaimer
    assert "safety_disclaimer" in data
    assert "simulated" in data["safety_disclaimer"].lower()


def test_frontend_dashboard_routes(client: TestClient):
    """
    Tests that the frontend dashboard routes return the static HTML page.
    """
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "Disaster UAV Command Center" in res_root.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "text/html" in res_dash.headers.get("content-type", "")
    assert "Disaster UAV Command Center" in res_dash.text



