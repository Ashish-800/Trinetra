"""
SQLAlchemy ORM Models for Disaster UAV System.
Entities:
- Incident
- MediaAsset
- Detection
- Track
- PriorityAssessment
- Resource
- ResourceRecommendation
"""
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(150), nullable=False)
    disaster_type: Mapped[str] = mapped_column(String(50), default="GENERAL", nullable=False)
    location_name: Mapped[str] = mapped_column(String(200), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    media_assets: Mapped[list["MediaAsset"]] = relationship(
        "MediaAsset", back_populates="incident", cascade="all, delete-orphan"
    )
    tracks: Mapped[list["Track"]] = relationship(
        "Track", back_populates="incident", cascade="all, delete-orphan"
    )


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    media_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "IMAGE" or "VIDEO"
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    incident: Mapped["Incident"] = relationship("Incident", back_populates="media_assets")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="media_asset", cascade="all, delete-orphan"
    )


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    track_label: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    start_time: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    incident: Mapped["Incident"] = relationship("Incident", back_populates="tracks")
    detections: Mapped[list["Detection"]] = relationship("Detection", back_populates="track")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    media_asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=True, index=True
    )
    frame_id: Mapped[int | None] = mapped_column(
        ForeignKey("aerial_frames.id", ondelete="CASCADE"), nullable=True, index=True
    )
    track_id: Mapped[int | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    tracking_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    class_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Bounding Box Coordinates (normalized [0,1] or pixel coordinates)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    # Frame reference
    frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_reference: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Coordinates & location source
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_radius_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_source: Mapped[str] = mapped_column(
        String(50), default="simulated", nullable=False
    )  # e.g. "simulated", "gps_exif", "manual_input", "unknown"
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Scoring & Prioritization
    urgency_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    uncertainty_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    priority_level: Mapped[str] = mapped_column(String(30), default="LOW", nullable=False, index=True)

    # Human Review Status
    review_status: Mapped[str] = mapped_column(
        String(30), default="PENDING_REVIEW", nullable=False, index=True
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    media_asset: Mapped["MediaAsset | None"] = relationship("MediaAsset", back_populates="detections")
    frame: Mapped["AerialFrame | None"] = relationship("AerialFrame", back_populates="detections")  # noqa: F821
    track: Mapped["Track | None"] = relationship("Track", back_populates="detections")
    priority_assessments: Mapped[list["PriorityAssessment"]] = relationship(
        "PriorityAssessment", back_populates="detection", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["ResourceRecommendation"]] = relationship(
        "ResourceRecommendation", back_populates="detection", cascade="all, delete-orphan"
    )
    review_logs: Mapped[list["HumanReviewLog"]] = relationship(  # noqa: F821
        "HumanReviewLog", back_populates="detection", cascade="all, delete-orphan"
    )

    @property
    def bounding_box(self) -> dict:
        return {"x1": self.bbox_x1, "y1": self.bbox_y1, "x2": self.bbox_x2, "y2": self.bbox_y2}

    @property
    def bbox(self) -> dict:
        return {"x1": self.bbox_x1, "y1": self.bbox_y1, "x2": self.bbox_x2, "y2": self.bbox_y2}


class PriorityAssessment(Base):
    __tablename__ = "priority_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    urgency_score: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty_score: Mapped[float] = mapped_column(Float, nullable=False)
    priority_level: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    assessed_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    detection: Mapped["Detection"] = relationship("Detection", back_populates="priority_assessments")


class Resource(Base):
    __tablename__ = "resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    total_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    available_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    simulated_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    simulated_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    recommendations: Mapped[list["ResourceRecommendation"]] = relationship(
        "ResourceRecommendation", back_populates="resource"
    )


class ResourceRecommendation(Base):
    __tablename__ = "resource_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resource_id: Mapped[int] = mapped_column(
        ForeignKey("resources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    suitability_score: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_distance_m: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="PENDING_REVIEW", nullable=False)
    requires_human_authorization: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    detection: Mapped["Detection"] = relationship("Detection", back_populates="recommendations")
    resource: Mapped["Resource"] = relationship("Resource", back_populates="recommendations")
