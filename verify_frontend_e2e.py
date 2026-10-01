"""
Frontend and Multimodal API End-to-End Verification.
Tests the real multimodal endpoint with an actual VisDrone validation image,
verifying that the response conforms to the frontend dashboard requirements.
"""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app

def test_frontend_multimodal_e2e():
    client = TestClient(app)

    # 1. Verify dashboard HTML is served
    dash_res = client.get("/")
    assert dash_res.status_code == 200
    assert "Disaster UAV Command Center" in dash_res.text
    assert "uav-canvas" in dash_res.text
    print("[PASS] Dashboard HTML served at /")

    # 2. Verify static assets are served
    css_res = client.get("/static/css/style.css")
    assert css_res.status_code == 200
    assert "command center" in css_res.text.lower() or "--color-bg" in css_res.text
    print("[PASS] Static CSS served at /static/css/style.css")

    js_res = client.get("/static/js/app.js")
    assert js_res.status_code == 200
    assert "drawBoundingBoxes" in js_res.text
    assert "executeMultimodalAnalysis" in js_res.text
    print("[PASS] Static JS served at /static/js/app.js")

    # 3. Test with real VisDrone image
    val_img_path = Path("archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg")
    assert val_img_path.exists(), f"Image not found at {val_img_path}"

    with open(val_img_path, "rb") as f:
        file_bytes = f.read()

    print(f"Uploading real VisDrone image: {val_img_path.name} ({len(file_bytes)} bytes)...")
    res = client.post(
        "/api/v1/analysis/multimodal",
        data={"enable_segmentation": "true"},
        files={"file": (val_img_path.name, file_bytes, "image/jpeg")},
    )

    assert res.status_code == 200, f"API error: {res.status_code} - {res.text}"
    data = res.json()

    print("\n--- Multimodal API Response Summary ---")
    print(f"Incident ID: {data.get('incident_id')}")
    print(f"Media Asset ID: {data.get('media_asset_id')}")
    print(f"Total Detections Found: {data.get('total_detections_found')}")
    print(f"Unique Person Tracks: {data.get('unique_person_tracks')}")
    print(f"Device Info: {data.get('device_info')}")

    # Verify Frontend fields
    assert "incident" in data and data["incident"]["id"] == data["incident_id"]
    assert "media" in data and data["media"]["id"] == data["media_asset_id"]
    assert "detections" in data
    assert "priority_assessment" in data
    assert "recommendation_outcome" in data
    assert "scene_context" in data
    assert "safety_disclaimer" in data

    # Check safe terminology
    for det in data["detections"]:
        assert "survivor" not in det["label"].lower() or "potential" in det["label"].lower()
        assert det["location_source"] == "simulated"
        assert det["review_status"] == "PENDING_REVIEW"
        assert "bbox" in det
        assert det["confidence"] >= 0.0

    # Check advisory recommendations
    rec_outcome = data["recommendation_outcome"]
    for rec in rec_outcome.get("recommendations", []):
        assert rec["requires_human_authorization"] is True
        assert rec["status"] == "PENDING_REVIEW"

    # Check segmentation context
    scene = data["scene_context"]
    assert "water_coverage_pct" in scene
    assert "debris_coverage_pct" in scene
    assert "road_coverage_pct" in scene
    assert "tree_coverage_pct" in scene
    assert "inferred_hazard_severity" in scene
    assert "inferred_accessibility" in scene

    print("\n[SUCCESS] Multimodal API and Frontend integration fully validated!")

if __name__ == "__main__":
    test_frontend_multimodal_e2e()
