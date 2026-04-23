import datetime
import uuid
from sqlalchemy import String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="user")  # "user" or "pro"
    player_name: Mapped[str | None] = mapped_column(String(100), nullable=True)  # pro player name
    team: Mapped[str | None] = mapped_column(String(100), nullable=True)  # pro team
    agent: Mapped[str | None] = mapped_column(String(50), nullable=True)  # Valorant agent
    map_name: Mapped[str | None] = mapped_column(String(50), nullable=True)  # map played
    rank: Mapped[str | None] = mapped_column(String(50), nullable=True)  # rank tier
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)  # free-form tags

    # File info
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Analysis link (optional - if this VOD has been analyzed)
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), default="uploaded")  # uploaded, analyzing, ready, failed

    # Progress tracking while the pro-VOD analysis runs
    analysis_progress: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analysis_status_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
