"""
Unit tests for Media Ingestion Service (Phase 3).
Verifies file validation, error handling, image ingestion, and video frame extraction
using dynamically created synthetic test assets (zero internet downloads).
"""
from pathlib import Path
import cv2
import numpy as np
import pytest
from app.services.ingestion import (
    CorruptedMediaError,
    MediaIngestionService,
    MediaNotFoundError,
    UnsupportedMediaFormatError,
)


@pytest.fixture
def temp_media_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory containing synthetic media assets."""
    media_dir = tmp_path / "synthetic_media"
    media_dir.mkdir()
    return media_dir


@pytest.fixture
def synthetic_image_path(temp_media_dir: Path) -> Path:
    """Generates a small 120x80 synthetic RGB image for testing."""
    img_path = temp_media_dir / "sample_aerial.jpg"
    # Create a 3-channel RGB image (80 height x 120 width)
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[:] = (180, 120, 70)  # Terrain-like brownish/grey color
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def synthetic_video_path(temp_media_dir: Path) -> Path:
    """Generates a tiny 1-second (10 frames at 10 fps) synthetic video."""
    video_path = temp_media_dir / "sample_flight.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    width, height = 64, 64
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for frame_idx in range(10):
        frame = np.full((height, width, 3), frame_idx * 20, dtype=np.uint8)
        out.write(frame)
    out.release()

    return video_path


def test_validate_nonexistent_file(tmp_path: Path):
    service = MediaIngestionService(default_output_dir=tmp_path / "out")
    with pytest.raises(MediaNotFoundError) as exc_info:
        service.ingest(tmp_path / "does_not_exist.jpg")
    assert "not found" in str(exc_info.value)


def test_validate_unsupported_format(temp_media_dir: Path, tmp_path: Path):
    bad_file = temp_media_dir / "flight_log.txt"
    bad_file.write_text("dummy flight telemetry")

    service = MediaIngestionService(default_output_dir=tmp_path / "out")
    with pytest.raises(UnsupportedMediaFormatError) as exc_info:
        service.ingest(bad_file)
    assert "Unsupported media format" in str(exc_info.value)


def test_validate_empty_corrupted_file(temp_media_dir: Path, tmp_path: Path):
    empty_file = temp_media_dir / "empty.jpg"
    empty_file.touch()

    service = MediaIngestionService(default_output_dir=tmp_path / "out")
    with pytest.raises(CorruptedMediaError) as exc_info:
        service.ingest(empty_file)
    assert "empty" in str(exc_info.value).lower()


def test_validate_fake_image_file(temp_media_dir: Path, tmp_path: Path):
    corrupt_file = temp_media_dir / "corrupted.png"
    corrupt_file.write_bytes(b"THIS_IS_NOT_A_VALID_PNG_HEADER_12345")

    service = MediaIngestionService(default_output_dir=tmp_path / "out")
    with pytest.raises(CorruptedMediaError) as exc_info:
        service.ingest(corrupt_file)
    assert "Failed to decode" in str(exc_info.value)


def test_ingest_valid_image(synthetic_image_path: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_frames"
    service = MediaIngestionService(default_output_dir=out_dir)

    result = service.ingest(synthetic_image_path, custom_output_dir=out_dir)

    assert result.media_type == "IMAGE"
    assert result.width_px == 120
    assert result.height_px == 80
    assert result.total_frames_extracted == 1
    assert len(result.frames) == 1

    extracted = result.frames[0]
    assert extracted.frame_index == 0
    assert extracted.timestamp_seconds == 0.0
    assert Path(extracted.output_path).exists()


def test_ingest_valid_video_and_frame_sampling(synthetic_video_path: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_video_frames"
    service = MediaIngestionService(default_output_dir=out_dir)

    # 10 fps video, 10 frames total (1.0 sec duration).
    # Sample interval 0.5s -> extracts at frame 0 and frame 5.
    result = service.ingest(synthetic_video_path, interval_seconds=0.5, custom_output_dir=out_dir)

    assert result.media_type == "VIDEO"
    assert result.width_px == 64
    assert result.height_px == 64
    assert result.duration_seconds == 1.0
    assert result.total_frames_extracted >= 2

    # Check extracted frame properties
    first_frame = result.frames[0]
    assert first_frame.frame_index == 0
    assert first_frame.timestamp_seconds == 0.0
    assert Path(first_frame.output_path).exists()

    second_frame = result.frames[1]
    assert second_frame.frame_index == 5
    assert second_frame.timestamp_seconds == 0.5
    assert Path(second_frame.output_path).exists()
