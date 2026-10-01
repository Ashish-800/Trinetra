"""
Media Ingestion Service.
Handles local image and video file validation, frame extraction,
and metadata preservation without loading entire videos into memory.
Enforces defensive caps on duration, frame count, and path safety.
"""
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Optional
import cv2
from pydantic import BaseModel, Field

from app.core.config import settings


class MediaIngestionError(Exception):
    """Base exception for media ingestion failures."""
    pass


class MediaNotFoundError(MediaIngestionError):
    """Raised when the specified file path does not exist."""
    pass


class UnsupportedMediaFormatError(MediaIngestionError):
    """Raised when file extension or MIME type is not supported."""
    pass


class CorruptedMediaError(MediaIngestionError):
    """Raised when media file cannot be decoded or is corrupt."""
    pass


class ExcessiveMediaResourceError(MediaIngestionError):
    """Raised when media exceeds defensive operational duration or frame limits."""
    pass


class ExtractedFrame(BaseModel):
    """Metadata for an individual extracted video or image frame."""
    frame_index: int = Field(..., ge=0, description="Sequential index of the frame in the source video (0 for images)")
    timestamp_seconds: float = Field(..., ge=0.0, description="Timestamp offset in seconds from video start")
    output_path: str = Field(..., description="Absolute or relative file path to the saved frame image")
    width_px: int = Field(..., gt=0)
    height_px: int = Field(..., gt=0)


class IngestionResult(BaseModel):
    """Overall summary of ingested asset and its extracted frames."""
    source_file_path: str
    media_type: str = Field(..., pattern="^(IMAGE|VIDEO)$")
    mime_type: str
    file_size_bytes: int
    width_px: int
    height_px: int
    duration_seconds: Optional[float] = None
    total_frames_extracted: int
    frames: List[ExtractedFrame]


class MediaIngestionService:
    """Service to safely ingest, validate, and extract frames from aerial media."""

    SUPPORTED_IMAGE_EXTENSIONS = settings.ALLOWED_IMAGE_EXTENSIONS
    SUPPORTED_VIDEO_EXTENSIONS = settings.ALLOWED_VIDEO_EXTENSIONS

    def __init__(self, default_output_dir: Optional[Path] = None):
        self.output_dir = default_output_dir or Path(settings.UPLOAD_DIR / "extracted_frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename_stem(stem: str) -> str:
        """Sanitizes file stem to eliminate path traversal and hazardous characters."""
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
        return clean[:64] or "unnamed_media"

    def validate_file(self, file_path: Path) -> tuple[str, str]:
        """
        Validates file existence and supported extension.
        Returns (media_type, mime_type) where media_type is 'IMAGE' or 'VIDEO'.
        """
        if not file_path.exists() or not file_path.is_file():
            raise MediaNotFoundError(f"Media file not found at path: {file_path}")

        ext = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            return "IMAGE", mime_type or f"image/{ext.lstrip('.')}"
        elif ext in self.SUPPORTED_VIDEO_EXTENSIONS:
            return "VIDEO", mime_type or f"video/{ext.lstrip('.')}"
        else:
            supported = sorted(list(self.SUPPORTED_IMAGE_EXTENSIONS | self.SUPPORTED_VIDEO_EXTENSIONS))
            raise UnsupportedMediaFormatError(
                f"Unsupported media format '{ext}'. Supported formats: {', '.join(supported)}"
            )

    def ingest(
        self,
        file_path: str | Path,
        interval_seconds: float = 1.0,
        custom_output_dir: Optional[Path] = None,
    ) -> IngestionResult:
        """
        Ingests a media file (image or video), validates it, and extracts frames.
        Enforces defensive sampling and size boundaries.
        """
        if interval_seconds < settings.MIN_SAMPLING_INTERVAL_SECONDS:
            raise ExcessiveMediaResourceError(
                f"Sampling interval ({interval_seconds}s) is too low. "
                f"Minimum allowed interval is {settings.MIN_SAMPLING_INTERVAL_SECONDS}s to prevent frame explosion."
            )

        path = Path(file_path).resolve()
        media_type, mime_type = self.validate_file(path)
        file_size = path.stat().st_size

        if file_size == 0:
            raise CorruptedMediaError(f"Media file is empty (0 bytes): {path}")

        clean_stem = self.sanitize_filename_stem(path.stem)
        out_dir = custom_output_dir or (self.output_dir / clean_stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        if media_type == "IMAGE":
            return self._process_image(path, mime_type, file_size, out_dir, clean_stem)
        else:
            return self._process_video(path, mime_type, file_size, out_dir, interval_seconds)

    def _process_image(
        self, path: Path, mime_type: str, file_size: int, out_dir: Path, clean_stem: str
    ) -> IngestionResult:
        """Processes a single static aerial image."""
        img = cv2.imread(str(path))
        if img is None:
            raise CorruptedMediaError(f"Failed to decode image file: {path}. File may be corrupted or invalid.")

        height, width = img.shape[:2]
        output_frame_path = out_dir / f"{clean_stem}_frame_0000.jpg"
        cv2.imwrite(str(output_frame_path), img)

        frame = ExtractedFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            output_path=str(output_frame_path),
            width_px=width,
            height_px=height,
        )

        return IngestionResult(
            source_file_path=str(path),
            media_type="IMAGE",
            mime_type=mime_type,
            file_size_bytes=file_size,
            width_px=width,
            height_px=height,
            duration_seconds=0.0,
            total_frames_extracted=1,
            frames=[frame],
        )

    def _process_video(
        self,
        path: Path,
        mime_type: str,
        file_size: int,
        out_dir: Path,
        interval_seconds: float,
    ) -> IngestionResult:
        """
        Streams a video file frame-by-frame using cv2.VideoCapture,
        avoiding loading the entire video into RAM.
        Enforces MAX_VIDEO_DURATION_SECONDS and MAX_FRAMES_PER_VIDEO defensive limits.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise CorruptedMediaError(f"Failed to open video file: {path}. Codec or container may be unreadable.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0.0 or total_video_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise CorruptedMediaError(
                f"Video stream headers are invalid (fps={fps}, total_frames={total_video_frames}, dims={width}x{height})"
            )

        duration = total_video_frames / fps
        if duration > settings.MAX_VIDEO_DURATION_SECONDS:
            cap.release()
            raise ExcessiveMediaResourceError(
                f"Video duration ({duration:.1f}s) exceeds maximum permitted limit of {settings.MAX_VIDEO_DURATION_SECONDS}s."
            )

        frame_step = max(1, int(round(fps * interval_seconds)))
        extracted_frames: List[ExtractedFrame] = []
        current_frame_idx = 0

        try:
            while True:
                # Enforce defensive frame extraction cap
                if len(extracted_frames) >= settings.MAX_FRAMES_PER_VIDEO:
                    break

                ret = cap.grab()
                if not ret:
                    break

                if current_frame_idx % frame_step == 0:
                    ret_decode, frame = cap.retrieve()
                    if ret_decode and frame is not None:
                        timestamp = round(current_frame_idx / fps, 3)
                        out_frame_filename = f"frame_{current_frame_idx:06d}_{timestamp:.2f}s.jpg"
                        out_path = out_dir / out_frame_filename
                        cv2.imwrite(str(out_path), frame)

                        extracted_frames.append(
                            ExtractedFrame(
                                frame_index=current_frame_idx,
                                timestamp_seconds=timestamp,
                                output_path=str(out_path),
                                width_px=width,
                                height_px=height,
                            )
                        )

                current_frame_idx += 1

        finally:
            cap.release()

        if not extracted_frames:
            raise CorruptedMediaError(f"Video file contained no readable frames: {path}")

        return IngestionResult(
            source_file_path=str(path),
            media_type="VIDEO",
            mime_type=mime_type,
            file_size_bytes=file_size,
            width_px=width,
            height_px=height,
            duration_seconds=round(duration, 3),
            total_frames_extracted=len(extracted_frames),
            frames=extracted_frames,
        )
