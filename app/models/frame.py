"""
SQLAlchemy ORM Model for Aerial Frames.
"""
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class AerialFrame(Base):
    __tablename__ = "aerial_frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    mission_id: Mapped[int] = mapped_column(ForeignKey("missions.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    height_px: Mapped[int] = mapped_column(Integer, nullable=False)

    # Telemetry
    uav_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    uav_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    altitude_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    gimbal_pitch_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    gimbal_yaw_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    focal_length_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    fov_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    captured_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    mission: Mapped["Mission"] = relationship("Mission", back_populates="frames")  # noqa: F821
    detections: Mapped[list["Detection"]] = relationship(  # noqa: F821
        "Detection", back_populates="frame", cascade="all, delete-orphan"
    )
