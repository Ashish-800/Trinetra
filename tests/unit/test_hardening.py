"""
Automated Unit Tests for Phase 12 Hardening and Defensive Boundaries.
Tests defensive limits, resource caps, path sanitization, log masking, and error handling.
"""
import logging
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.core.logging import SensitiveDataLogFilter
from app.main import app
from app.services.ingestion import (
    ExcessiveMediaResourceError,
    MediaIngestionService,
)


@pytest.fixture
def test_video_file(tmp_path: Path) -> Path:
    """Creates a short 5-frame synthetic MP4 video."""
    vid_path = tmp_path / "short_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))
    for i in range(5):
        frame = np.full((64, 64, 3), 100 + i * 20, dtype=np.uint8)
        out.write(frame)
    out.release()
    return vid_path


def test_sampling_interval_too_low_rejected(test_video_file: Path, tmp_path: Path):
    """Verifies that dangerously small sampling intervals (< 0.25s) are rejected."""
    service = MediaIngestionService(default_output_dir=tmp_path / "out")
    with pytest.raises(ExcessiveMediaResourceError) as excinfo:
        service.ingest(test_video_file, interval_seconds=0.05)
    assert "Minimum allowed interval" in str(excinfo.value)


def test_filename_stem_sanitization():
    """Verifies that hazardous path traversal characters are scrubbed from file stems."""
    unsafe_stem = "../../etc/passwd..\\sneaky*name#?"
    safe_stem = MediaIngestionService.sanitize_filename_stem(unsafe_stem)
    assert ".." not in safe_stem
    assert "/" not in safe_stem
    assert "\\" not in safe_stem
    assert safe_stem.replace("_", "").isalnum()


def test_frame_extraction_cap_enforced(tmp_path: Path):
    """Verifies that frame extraction strictly honors MAX_FRAMES_PER_VIDEO cap."""
    # Temporarily set max frames to 3
    original_cap = settings.MAX_FRAMES_PER_VIDEO
    settings.MAX_FRAMES_PER_VIDEO = 3

    vid_path = tmp_path / "multi_frame.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))
    # Write 20 frames
    for i in range(20):
        frame = np.full((64, 64, 3), 50, dtype=np.uint8)
        out.write(frame)
    out.release()

    service = MediaIngestionService(default_output_dir=tmp_path / "frames")
    result = service.ingest(vid_path, interval_seconds=0.1)  # would extract 10 frames without cap

    settings.MAX_FRAMES_PER_VIDEO = original_cap
    assert result.total_frames_extracted == 3
    assert len(result.frames) == 3


def test_sensitive_data_log_filter():
    """Verifies that SensitiveDataLogFilter scrubs tokens and host user paths."""
    filt = SensitiveDataLogFilter()

    # Test token masking
    rec1 = logging.LogRecord("test", logging.INFO, "test.py", 10, "Bearer abc123def456xyz==", (), None)
    filt.filter(rec1)
    assert "Bearer [FILTERED_TOKEN]" in rec1.msg
    assert "abc123def456xyz" not in rec1.msg

    # Test user directory masking
    rec2 = logging.LogRecord("test", logging.INFO, "test.py", 10, "Accessing file at C:\\Users\\JohnDoe\\secret.jpg", (), None)
    filt.filter(rec2)
    assert "C:\\Users\\[USER]\\" in rec2.msg
    assert "JohnDoe" not in rec2.msg


def test_api_oversized_upload_rejected(db_session: Session, tmp_path: Path):
    """Verifies that uploads exceeding MAX_UPLOAD_SIZE_BYTES return HTTP 413."""
    # Temporarily lower upload limit to 100 KB for testing
    from app.api.v1.endpoints import media
    original_limit = media.MAX_UPLOAD_SIZE_BYTES
    media.MAX_UPLOAD_SIZE_BYTES = 100 * 1024  # 100 KB

    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app) as client:
        # Create incident
        inc_res = client.post("/api/v1/incidents/", json={"title": "Test Sector", "location_name": "Grid 1"})
        inc_id = inc_res.json()["id"]

        # Attempt to upload 200 KB dummy file
        big_data = b"X" * (200 * 1024)
        upload_res = client.post(
            "/api/v1/media/upload",
            data={"incident_id": inc_id},
            files={"file": ("big_image.jpg", big_data, "image/jpeg")},
        )
        assert upload_res.status_code == 413
        assert "exceeds maximum allowed size" in upload_res.json()["detail"]

    media.MAX_UPLOAD_SIZE_BYTES = original_limit
    app.dependency_overrides.clear()
