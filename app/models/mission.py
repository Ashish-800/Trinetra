"""
SQLAlchemy ORM Model for Flight Missions.
"""
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Mission(Base):
    __tablename__ = "missions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    disaster_type: Mapped[str] = mapped_column(String(50), default="GENERAL", nullable=False)
    location_name: Mapped[str] = mapped_column(String(200), nullable=False)
    simulated_base_lat: Mapped[float] = mapped_column(Float, nullable=False)
    simulated_base_lon: Mapped[float] = mapped_column(Float, nullable=False)
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
    frames: Mapped[list["AerialFrame"]] = relationship(  # noqa: F821
        "AerialFrame", back_populates="mission", cascade="all, delete-orphan"
    )
