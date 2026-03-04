"""
Knowledge Base API Router.

Endpoints for managing the AI's accumulated tactical knowledge.
Knowledge entries are created automatically from analyses and can
also be added manually.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.database import get_db
from app.models.knowledge import KnowledgeEntry

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


# ── Schemas ──────────────────────────────────────────────────────────

class KnowledgeCreate(BaseModel):
    source_type: str = "manual"
    source_id: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    rank: Optional[str] = None
    title: str
    description: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    confidence: float = 0.5
    tags: Optional[list[str]] = None


class KnowledgeResponse(BaseModel):
    id: str
    source_type: str
    source_id: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    rank: Optional[str] = None
    title: str
    description: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    confidence: float
    observation_count: int
    tags: Optional[list[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class KnowledgeListItem(BaseModel):
    id: str
    source_type: str
    category: str
    subcategory: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    title: str
    confidence: float
    observation_count: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("", response_model=KnowledgeResponse, status_code=201)
async def create_knowledge_entry(
    entry: KnowledgeCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge entry manually."""
    knowledge = KnowledgeEntry(
        id=str(uuid.uuid4()),
        source_type=entry.source_type,
        source_id=entry.source_id,
        category=entry.category,
        subcategory=entry.subcategory,
        agent=entry.agent,
        map_name=entry.map_name,
        rank=entry.rank,
        title=entry.title,
        description=entry.description,
        metric_name=entry.metric_name,
        metric_value=entry.metric_value,
        confidence=entry.confidence,
        tags=entry.tags,
    )
    db.add(knowledge)
    await db.commit()
    await db.refresh(knowledge)
    return KnowledgeResponse.model_validate(knowledge)


@router.get("", response_model=list[KnowledgeListItem])
async def list_knowledge(
    category: str | None = None,
    agent: str | None = None,
    map_name: str | None = None,
    source_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List knowledge entries with optional filters."""
    query = select(KnowledgeEntry).order_by(desc(KnowledgeEntry.confidence), desc(KnowledgeEntry.observation_count))

    if category:
        query = query.where(KnowledgeEntry.category == category)
    if agent:
        query = query.where(KnowledgeEntry.agent == agent)
    if map_name:
        query = query.where(KnowledgeEntry.map_name == map_name)
    if source_type:
        query = query.where(KnowledgeEntry.source_type == source_type)

    result = await db.execute(query)
    entries = result.scalars().all()
    return [KnowledgeListItem.model_validate(e) for e in entries]


@router.get("/stats")
async def knowledge_stats(
    db: AsyncSession = Depends(get_db),
):
    """Summary statistics for the knowledge base."""
    result = await db.execute(select(KnowledgeEntry))
    entries = result.scalars().all()

    total = len(entries)
    categories: dict[str, int] = {}
    agents: dict[str, int] = {}
    maps: dict[str, int] = {}
    sources: dict[str, int] = {}

    for e in entries:
        categories[e.category] = categories.get(e.category, 0) + 1
        sources[e.source_type] = sources.get(e.source_type, 0) + 1
        if e.agent:
            agents[e.agent] = agents.get(e.agent, 0) + 1
        if e.map_name:
            maps[e.map_name] = maps.get(e.map_name, 0) + 1

    avg_confidence = sum(e.confidence for e in entries) / total if total else 0

    return {
        "total_entries": total,
        "avg_confidence": round(avg_confidence, 3),
        "categories": categories,
        "agents": agents,
        "maps": maps,
        "sources": sources,
    }


@router.get("/{entry_id}", response_model=KnowledgeResponse)
async def get_knowledge_entry(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single knowledge entry by ID."""
    result = await db.execute(select(KnowledgeEntry).where(KnowledgeEntry.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    return KnowledgeResponse.model_validate(entry)


@router.delete("/{entry_id}")
async def delete_knowledge_entry(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a knowledge entry."""
    result = await db.execute(select(KnowledgeEntry).where(KnowledgeEntry.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")

    await db.delete(entry)
    await db.commit()
    return {"message": "Knowledge entry deleted"}


@router.post("/extract-from-analysis/{analysis_id}")
async def extract_knowledge_from_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Extract knowledge entries from a completed analysis.

    This reads the analysis results and creates knowledge entries
    for patterns, benchmarks, and insights found.
    """
    from app.models.analysis import Analysis

    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.status != "completed":
        raise HTTPException(status_code=400, detail="Analysis must be completed before extracting knowledge")

    entries_created = []

    # Extract crosshair insights
    if analysis.crosshair_score is not None:
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            source_type="analysis",
            source_id=analysis_id,
            category="crosshair",
            subcategory="head_level",
            title=f"Crosshair score: {analysis.crosshair_score}/100",
            description=f"Analysis of {analysis.filename} yielded crosshair score {analysis.crosshair_score}. "
                        f"{'Excellent' if analysis.crosshair_score >= 80 else 'Good' if analysis.crosshair_score >= 60 else 'Needs improvement'} "
                        f"head-level consistency.",
            metric_name="crosshair_score",
            metric_value=float(analysis.crosshair_score),
            confidence=0.7,
            tags=["auto-extracted"],
        )
        db.add(entry)
        entries_created.append(entry.title)

    # Extract movement insights
    if analysis.movement_score is not None:
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            source_type="analysis",
            source_id=analysis_id,
            category="movement",
            subcategory="counter_strafe",
            title=f"Movement score: {analysis.movement_score}/100",
            description=f"Analysis of {analysis.filename} yielded movement score {analysis.movement_score}. "
                        f"{'Excellent' if analysis.movement_score >= 80 else 'Good' if analysis.movement_score >= 60 else 'Needs improvement'} "
                        f"counter-strafing and peek mechanics.",
            metric_name="movement_score",
            metric_value=float(analysis.movement_score),
            confidence=0.7,
            tags=["auto-extracted"],
        )
        db.add(entry)
        entries_created.append(entry.title)

    # Extract decision insights
    if analysis.decision_score is not None:
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            source_type="analysis",
            source_id=analysis_id,
            category="tactical",
            subcategory="decision_making",
            title=f"Decision score: {analysis.decision_score}/100",
            description=f"Analysis of {analysis.filename} yielded decision score {analysis.decision_score}. "
                        f"{'Excellent' if analysis.decision_score >= 80 else 'Good' if analysis.decision_score >= 60 else 'Needs improvement'} "
                        f"tactical decision-making.",
            metric_name="decision_score",
            metric_value=float(analysis.decision_score),
            confidence=0.7,
            tags=["auto-extracted"],
        )
        db.add(entry)
        entries_created.append(entry.title)

    # Extract positioning insights
    if analysis.map_score is not None:
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            source_type="analysis",
            source_id=analysis_id,
            category="positioning",
            subcategory="map_awareness",
            title=f"Map/Positioning score: {analysis.map_score}/100",
            description=f"Analysis of {analysis.filename} yielded positioning score {analysis.map_score}. "
                        f"{'Excellent' if analysis.map_score >= 80 else 'Good' if analysis.map_score >= 60 else 'Needs improvement'} "
                        f"map awareness and rotations.",
            metric_name="map_score",
            metric_value=float(analysis.map_score),
            confidence=0.7,
            tags=["auto-extracted"],
        )
        db.add(entry)
        entries_created.append(entry.title)

    # Extract overall benchmark
    if analysis.overall_score is not None:
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            source_type="analysis",
            source_id=analysis_id,
            category="benchmark",
            subcategory="overall",
            title=f"Overall performance: {analysis.overall_score}/100",
            description=f"Overall gameplay analysis of {analysis.filename}: score {analysis.overall_score}/100. "
                        f"This data point contributes to the performance baseline for future comparisons.",
            metric_name="overall_score",
            metric_value=float(analysis.overall_score),
            confidence=0.8,
            tags=["auto-extracted", "benchmark"],
        )
        db.add(entry)
        entries_created.append(entry.title)

    await db.commit()

    return {
        "analysis_id": analysis_id,
        "entries_created": len(entries_created),
        "entries": entries_created,
    }
