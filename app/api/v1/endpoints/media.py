"""
Media Upload and Ingestion Endpoints.
Provides secure media upload without exposing arbitrary server filesystem paths.
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_ingestion_service
from app.core.config import settings
from app.models.domain import Incident, MediaAsset
from app.schemas.domain import MediaAssetRead
from app.services.ingestion import (
    CorruptedMediaError,
    MediaIngestionService,
    UnsupportedMediaFormatError,
)

router = APIRouter(prefix="/media", tags=["Media Ingestion"])

# Defensive maximum allowed file upload size (configurable via Settings)
MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_BYTES


@router.post("/upload", response_model=MediaAssetRead, status_code=status.HTTP_201_CREATED)
async def upload_media(
    incident_id: int = Form(..., description="ID of the incident to associate media with"),
    file: UploadFile = File(..., description="Aerial image or video file"),
    db: Session = Depends(get_db),
    ingestion: MediaIngestionService = Depends(get_ingestion_service),
) -> MediaAssetRead:
    """
    Securely uploads aerial imagery or video to the managed storage vault.
    Validates file format, size limits, and associates with a valid incident.
    Does not accept arbitrary client filesystem paths.
    """
    # 1. Verify incident exists
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} does not exist."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    # 2. Check extension
    file_ext = Path(file.filename).suffix.lower()
    allowed_exts = ingestion.SUPPORTED_IMAGE_EXTENSIONS | ingestion.SUPPORTED_VIDEO_EXTENSIONS
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}"
        )

    # 3. Create sanitized destination path in managed UPLOAD_DIR
    safe_name = f"inc_{incident_id}_{uuid.uuid4().hex[:8]}{file_ext}"
    destination_path = settings.UPLOAD_DIR / safe_name

    # 4. Stream file and enforce maximum size
    total_bytes = 0
    oversized = False
    try:
        with open(destination_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    oversized = True
                    break
                buffer.write(chunk)
    except Exception as e:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    if oversized:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    if total_bytes == 0:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    # 5. Extract metadata and validate with OpenCV
    is_video = file_ext in ingestion.SUPPORTED_VIDEO_EXTENSIONS
    media_type = "VIDEO" if is_video else "IMAGE"
    mime_type = file.content_type or ("video/mp4" if is_video else "image/jpeg")

    width_px = None
    height_px = None
    duration_sec = None
    frame_count = None

    try:
        if is_video:
            cap = cv2.VideoCapture(str(destination_path))
            if not cap.isOpened():
                destination_path.unlink(missing_ok=True)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted or unreadable video file.")
            width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            duration_sec = float(frame_count / fps) if frame_count and fps > 0 else 0.0
            cap.release()
        else:
            img = cv2.imread(str(destination_path))
            if img is None:
                destination_path.unlink(missing_ok=True)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted or unreadable image file.")
            height_px, width_px = img.shape[:2]
            frame_count = 1
            duration_sec = 0.0
    except HTTPException:
        raise
    except Exception as e:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error inspecting media: {str(e)}"
        )

    # 6. Save MediaAsset record
    media_asset = MediaAsset(
        incident_id=incident_id,
        file_path=str(destination_path),
        media_type=media_type,
        mime_type=mime_type,
        file_size_bytes=total_bytes,
        width_px=width_px,
        height_px=height_px,
        duration_seconds=duration_sec,
        frame_count=frame_count,
    )
    db.add(media_asset)
    db.commit()
    db.refresh(media_asset)

    return media_asset


@router.get("/{media_asset_id}", response_model=MediaAssetRead)
def get_media_asset(
    media_asset_id: int,
    db: Session = Depends(get_db),
) -> MediaAssetRead:
    """Retrieve metadata of a registered media asset by ID."""
    asset = db.query(MediaAsset).filter(MediaAsset.id == media_asset_id).first()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MediaAsset with ID {media_asset_id} not found."
        )
    return asset
