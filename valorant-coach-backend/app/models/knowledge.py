import datetime
import uuid
from sqlalchemy import String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class KnowledgeEntry(Base):
    """Accumulated tactical knowledge from VOD analyses.

    Each entry stores a pattern, insight, or benchmark learned from
    analysing gameplay VODs.  The knowledge base grows with every
    analysis and is used to improve future recommendations.
    """

    __tablename__ = "knowledge_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Where the insight came from
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "analysis", "pro_benchmark", "manual"
    source_id: Mapped[str | None] = mapped_column(String(36), nullable=True)  # analysis_id if from analysis

    # Classification
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    # e.g. "crosshair", "movement", "positioning", "economy", "ability_usage", "tactical", "agent_specific"
    subcategory: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # e.g. "head_level", "counter_strafe", "smoke_placement"

    # Context
    agent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    map_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    rank: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # The actual insight
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Numeric data (for benchmarks and stats)
    metric_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Confidence / weight (increases as more analyses confirm this pattern)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    observation_count: Mapped[int] = mapped_column(Integer, default=1)

    # Tags for flexible querying
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
