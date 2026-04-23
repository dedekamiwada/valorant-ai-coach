"""
Dataset API Router.

Endpoints for uploading, listing, updating, and deleting reference VOD datasets.
Datasets can be user-uploaded or downloaded from pro player channels.

Also exposes ``POST /{dataset_id}/analyze`` which runs the video analysis
pipeline on a pro player's VOD and converts the results into textual
"strengths" persisted in the knowledge base.
"""

import os
import uuid
import shutil
import asyncio
from threading import Thread

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.database import get_db
from app.models.dataset import Dataset
from app.models.knowledge import KnowledgeEntry
from app.schemas.dataset import (
    DatasetResponse,
    DatasetListItem,
    DatasetUploadResponse,
    DatasetUpdate,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
# Per-dataset analysis artefacts live under /tmp to avoid consuming the
# 1GB persistent volume shared with the SQLite database.
DATASET_ANALYSIS_DIR = os.path.join("/tmp", "dataset_analysis")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(DATASET_ANALYSIS_DIR, exist_ok=True)


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default=""),
    source: str = Form(default="user"),
    player_name: str = Form(default=""),
    team: str = Form(default=""),
    agent: str = Form(default=""),
    map_name: str = Form(default=""),
    rank: str = Form(default=""),
    tags: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a VOD file to the reference dataset.

    Accepts video files (mp4, avi, mkv, mov, webm).
    Metadata (name, source, player, agent, etc.) is provided via form fields.
    """
    allowed_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    filename = file.filename or "video.mp4"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {ext}. Permitidos: {', '.join(allowed_extensions)}",
        )

    dataset_id = str(uuid.uuid4())

    # Parse tags from comma-separated string
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    dataset = Dataset(
        id=dataset_id,
        name=name,
        description=description or None,
        source=source,
        player_name=player_name or None,
        team=team or None,
        agent=agent or None,
        map_name=map_name or None,
        rank=rank or None,
        tags=tag_list,
        filename=filename,
        status="uploaded",
    )
    db.add(dataset)
    await db.commit()

    # Save file to persistent storage
    file_dir = os.path.join(DATASET_DIR, dataset_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, filename)

    try:
        CHUNK_SIZE = 1024 * 1024  # 1 MB
        total_bytes = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)

        # Update file size
        dataset.file_size_bytes = total_bytes
        await db.commit()

    except OSError as exc:
        # Clean up on failure
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir, ignore_errors=True)
        await db.delete(dataset)
        await db.commit()
        raise HTTPException(
            status_code=507,
            detail=f"Falha ao salvar arquivo: {exc}",
        )

    return DatasetUploadResponse(
        id=dataset_id,
        message="Dataset uploaded successfully.",
    )


@router.get("", response_model=list[DatasetListItem])
async def list_datasets(
    source: str | None = None,
    agent: str | None = None,
    map_name: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all datasets, optionally filtered by source, agent, or map."""
    query = select(Dataset).order_by(desc(Dataset.created_at))

    if source:
        query = query.where(Dataset.source == source)
    if agent:
        query = query.where(Dataset.agent == agent)
    if map_name:
        query = query.where(Dataset.map_name == map_name)

    result = await db.execute(query)
    datasets = result.scalars().all()
    return [DatasetListItem.model_validate(d) for d in datasets]


# NOTE: /stats/summary MUST be declared before /{dataset_id} to avoid
# FastAPI matching "stats" as a dataset_id path parameter.
@router.get("/stats/summary")
async def dataset_stats(
    db: AsyncSession = Depends(get_db),
):
    """Get summary statistics about the dataset collection."""
    result = await db.execute(select(Dataset))
    datasets = result.scalars().all()

    total = len(datasets)
    pro_count = sum(1 for d in datasets if d.source == "pro")
    user_count = sum(1 for d in datasets if d.source == "user")

    agents: dict[str, int] = {}
    maps: dict[str, int] = {}
    for d in datasets:
        if d.agent:
            agents[d.agent] = agents.get(d.agent, 0) + 1
        if d.map_name:
            maps[d.map_name] = maps.get(d.map_name, 0) + 1

    total_size = sum(d.file_size_bytes or 0 for d in datasets)
    total_duration = sum(d.duration_seconds or 0 for d in datasets)

    return {
        "total_datasets": total,
        "pro_datasets": pro_count,
        "user_datasets": user_count,
        "total_size_bytes": total_size,
        "total_duration_seconds": total_duration,
        "agents": agents,
        "maps": maps,
    }


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full dataset details by ID."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse.model_validate(dataset)


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    update: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update dataset metadata."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(dataset, key, value)

    await db.commit()
    await db.refresh(dataset)
    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a dataset and its associated files."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete files from persistent storage
    file_dir = os.path.join(DATASET_DIR, dataset_id)
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir, ignore_errors=True)

    await db.delete(dataset)
    await db.commit()

    return {"message": "Dataset deleted successfully"}


# ── Pro VOD analysis ─────────────────────────────────────────────────


def _run_pro_vod_analysis(
    dataset_id: str,
    video_path: str,
    output_dir: str,
) -> None:
    """Run the analysis pipeline on a pro's VOD and persist the generated
    knowledge entries. Executed in a background thread (mirrors
    ``run_analysis_sync`` in routers/analysis.py).
    """
    # Local imports keep module-level import cost low and avoid circulars.
    from app.database import async_session
    from app.models.analysis import Analysis
    from app.services.video_pipeline import process_video
    from app.services.pro_vod_analyzer import (
        ProMetadata,
        extract_pro_strengths,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _update_status(status: str, progress: int, text: str) -> None:
        async with async_session() as db:
            result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if not dataset:
                return
            dataset.status = status
            dataset.analysis_progress = progress
            dataset.analysis_status_text = text
            await db.commit()

    def progress_cb(pct: int, text: str = "") -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(
                _update_status("analyzing", pct, text or "Analisando..."),
                loop,
            )
            future.result(timeout=5.0)
        except Exception:
            pass

    async def _run() -> None:
        async with async_session() as db:
            result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if not dataset:
                return

            dataset.status = "analyzing"
            dataset.analysis_progress = 2
            dataset.analysis_status_text = "Iniciando análise do VOD do pro..."
            await db.commit()

            try:
                pipeline_result = await loop.run_in_executor(
                    None, process_video, video_path, output_dir, progress_cb,
                )

                # Persist a full Analysis row so the regular dashboard can
                # render the pro's VOD with timeline/heatmap/rounds.
                analysis_id = str(uuid.uuid4())
                analysis = Analysis(
                    id=analysis_id,
                    filename=dataset.filename,
                    status="completed",
                    progress=100,
                    status_text="Análise completa!",
                    duration_seconds=pipeline_result.duration_seconds,
                    resolution=pipeline_result.resolution,
                    fps=pipeline_result.fps,
                    total_frames_analyzed=pipeline_result.total_frames_analyzed,
                    overall_score=pipeline_result.overall_score,
                    crosshair_score=pipeline_result.crosshair_score,
                    movement_score=pipeline_result.movement_score,
                    decision_score=pipeline_result.decision_score,
                    communication_score=pipeline_result.communication_score,
                    map_score=pipeline_result.map_score,
                    crosshair_data=pipeline_result.crosshair_data,
                    movement_data=pipeline_result.movement_data,
                    decision_data=pipeline_result.decision_data,
                    communication_data=pipeline_result.communication_data,
                    map_data=pipeline_result.map_data,
                    timeline_events=pipeline_result.timeline_events,
                    recommendations=pipeline_result.recommendations,
                    heatmap_data=pipeline_result.heatmap_data,
                    round_analysis=pipeline_result.round_analysis,
                    pro_comparison=pipeline_result.pro_comparison,
                )
                db.add(analysis)

                # Extract and persist pro strengths as knowledge entries.
                meta = ProMetadata(
                    dataset_id=dataset_id,
                    player_name=dataset.player_name,
                    team=dataset.team,
                    agent=dataset.agent,
                    map_name=dataset.map_name,
                    rank=dataset.rank,
                )
                strengths = extract_pro_strengths(pipeline_result, meta)
                for s in strengths:
                    db.add(KnowledgeEntry(
                        id=str(uuid.uuid4()),
                        source_type=s["source_type"],
                        source_id=s["source_id"],
                        category=s["category"],
                        subcategory=s.get("subcategory"),
                        agent=s.get("agent"),
                        map_name=s.get("map_name"),
                        rank=s.get("rank"),
                        title=s["title"],
                        description=s["description"],
                        metric_name=s.get("metric_name"),
                        metric_value=s.get("metric_value"),
                        confidence=float(s.get("confidence", 0.7)),
                        tags=s.get("tags"),
                    ))

                dataset.status = "ready"
                dataset.analysis_id = analysis_id
                dataset.analysis_progress = 100
                dataset.analysis_status_text = (
                    f"{len(strengths)} pontos fortes extraídos"
                )
                dataset.duration_seconds = pipeline_result.duration_seconds
                await db.commit()

            except Exception as exc:
                result2 = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id),
                )
                dataset2 = result2.scalar_one_or_none()
                if dataset2:
                    dataset2.status = "failed"
                    dataset2.analysis_status_text = f"Falha: {exc}"
                    await db.commit()
            finally:
                # Clean up scratch directory to free disk space.
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir, ignore_errors=True)

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


@router.post("/{dataset_id}/analyze")
async def analyze_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Trigger the AI analysis pipeline on a dataset's VOD.

    Intended for ``source="pro"`` datasets — the pipeline output is
    converted into textual "strengths" and stored in the knowledge base so
    the app has written-down reference material about what the pro does well.
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status == "analyzing":
        raise HTTPException(
            status_code=409,
            detail="Este dataset já está sendo analisado.",
        )

    file_dir = os.path.join(DATASET_DIR, dataset_id)
    video_path = os.path.join(file_dir, dataset.filename)
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail="Arquivo do dataset não encontrado em disco.",
        )

    output_dir = os.path.join(DATASET_ANALYSIS_DIR, dataset_id)
    os.makedirs(output_dir, exist_ok=True)

    # Kick off analysis in the background; return immediately so the client
    # can poll the dataset's status field for progress.
    dataset.status = "analyzing"
    dataset.analysis_progress = 0
    dataset.analysis_status_text = "Preparando análise..."
    await db.commit()

    Thread(
        target=_run_pro_vod_analysis,
        args=(dataset_id, video_path, output_dir),
        daemon=True,
    ).start()

    return {
        "id": dataset_id,
        "message": "Análise iniciada. Acompanhe o progresso via GET /api/datasets/{id}/analysis-status.",
    }


@router.get("/{dataset_id}/analysis-status")
async def dataset_analysis_status(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Check the status of a pro-VOD analysis."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Count knowledge entries generated from this dataset (if any).
    knowledge_count: int | None = None
    if dataset.status == "ready":
        knowledge_result = await db.execute(
            select(KnowledgeEntry).where(
                KnowledgeEntry.source_type == "pro_vod",
                KnowledgeEntry.source_id == dataset_id,
            )
        )
        knowledge_count = len(knowledge_result.scalars().all())

    return {
        "id": dataset.id,
        "status": dataset.status,
        "progress": dataset.analysis_progress or 0,
        "status_text": dataset.analysis_status_text,
        "analysis_id": dataset.analysis_id,
        "knowledge_entries_count": knowledge_count,
    }
