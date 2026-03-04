import datetime
import uuid
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, processing, completed, failed
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Video metadata
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(20), nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_frames_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Scores (0-100)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    crosshair_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    movement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    communication_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    map_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Status text for progress tracking
    status_text: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Detailed results stored as JSON
    crosshair_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    movement_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    communication_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timeline_events: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommendations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    heatmap_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    round_analysis: Mapped[list | None] = mapped_column(JSON, nullable=True)
    pro_comparison: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    map_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
