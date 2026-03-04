"""
Dataset API Router.

Endpoints for uploading, listing, updating, and deleting reference VOD datasets.
Datasets can be user-uploaded or downloaded from pro player channels.
"""

import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import (
    DatasetResponse,
    DatasetListItem,
    DatasetUploadResponse,
    DatasetUpdate,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)


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
