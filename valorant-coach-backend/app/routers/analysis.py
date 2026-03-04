"""
Analysis API Router.

Endpoints for uploading VODs, checking processing status,
and retrieving analysis results.
"""

import os
import uuid
import shutil
import asyncio
from threading import Thread
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.database import get_db
from app.models.analysis import Analysis
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisListItem,
    UploadResponse,
)
from app.services.video_pipeline import process_video, PipelineResult

router = APIRouter(prefix="/api", tags=["analysis"])

DATA_DIR = os.environ.get("DATA_DIR", "/data")
# Store uploads & processing artefacts in /tmp so they don't consume the
# small persistent volume (1 GB) that holds the SQLite database.
UPLOAD_DIR = os.path.join("/tmp", "uploads")
PROCESSING_DIR = os.path.join("/tmp", "processing")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)


def _cleanup_files(analysis_id: str) -> None:
    """Remove uploaded video and processing artefacts for *analysis_id*."""
    for base in (UPLOAD_DIR, PROCESSING_DIR):
        path = os.path.join(base, analysis_id)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def run_analysis_sync(analysis_id: str, video_path: str, output_dir: str):
    """Run the video analysis pipeline synchronously in a background thread."""
    from app.database import async_session

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def update_progress(pct: int, text: str):
        """Update progress in the database."""
        async with async_session() as db:
            result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one_or_none()
            if analysis:
                analysis.progress = pct
                analysis.status_text = text
                await db.commit()

    def progress_cb(pct: int, text: str = ""):
        """Sync progress callback that updates DB in real-time.

        process_video runs inside ``loop.run_in_executor`` (a worker thread)
        while the event loop keeps running.  We schedule the async DB update
        on the running loop from the worker thread with
        ``asyncio.run_coroutine_threadsafe`` which returns a ``Future`` we can
        optionally wait on.
        """
        try:
            future = asyncio.run_coroutine_threadsafe(
                update_progress(pct, text), loop,
            )
            future.result(timeout=5.0)
        except Exception:
            pass  # Don't crash processing if progress update fails

    async def _run():
        async with async_session() as db:
            # Update status to processing
            result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one_or_none()
            if not analysis:
                return

            analysis.status = "processing"
            analysis.progress = 2
            analysis.status_text = "Iniciando análise..."
            await db.commit()

            try:
                # Run the pipeline in a thread pool so the event loop stays
                # responsive for progress_cb's async DB updates.
                pipeline_result = await loop.run_in_executor(
                    None, process_video, video_path, output_dir, progress_cb,
                )

                # Update analysis with results
                result2 = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
                analysis2 = result2.scalar_one_or_none()
                if not analysis2:
                    return

                analysis2.status = "completed"
                analysis2.progress = 100
                analysis2.status_text = "Análise completa!"
                analysis2.duration_seconds = pipeline_result.duration_seconds
                analysis2.resolution = pipeline_result.resolution
                analysis2.fps = pipeline_result.fps
                analysis2.total_frames_analyzed = pipeline_result.total_frames_analyzed
                analysis2.overall_score = pipeline_result.overall_score
                analysis2.crosshair_score = pipeline_result.crosshair_score
                analysis2.movement_score = pipeline_result.movement_score
                analysis2.decision_score = pipeline_result.decision_score
                analysis2.communication_score = pipeline_result.communication_score
                analysis2.map_score = pipeline_result.map_score
                analysis2.crosshair_data = pipeline_result.crosshair_data
                analysis2.movement_data = pipeline_result.movement_data
                analysis2.decision_data = pipeline_result.decision_data
                analysis2.communication_data = pipeline_result.communication_data
                analysis2.map_data = pipeline_result.map_data
                analysis2.timeline_events = pipeline_result.timeline_events
                analysis2.recommendations = pipeline_result.recommendations
                analysis2.heatmap_data = pipeline_result.heatmap_data
                analysis2.round_analysis = pipeline_result.round_analysis
                analysis2.pro_comparison = pipeline_result.pro_comparison
                await db.commit()

            except Exception as e:
                result3 = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
                analysis3 = result3.scalar_one_or_none()
                if analysis3:
                    analysis3.status = "failed"
                    analysis3.error_message = str(e)
                    await db.commit()
            finally:
                # Clean up temporary files to free disk space
                _cleanup_files(analysis_id)

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


@router.post("/upload", response_model=UploadResponse)
async def upload_vod(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a Valorant VOD for analysis.

    Accepts video files (mp4, avi, mkv, mov, webm).
    Processing begins immediately in the background.
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    filename = file.filename or "video.mp4"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create analysis record
    analysis_id = str(uuid.uuid4())
    analysis = Analysis(
        id=analysis_id,
        filename=filename,
        status="pending",
        progress=0,
    )
    db.add(analysis)
    await db.commit()

    # Save uploaded file – stream to disk in chunks to avoid loading
    # the entire video into memory (which causes OOM / "Network error"
    # on the client side for large files).
    video_dir = os.path.join(UPLOAD_DIR, analysis_id)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, filename)

    try:
        CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
    except OSError as exc:
        # Disk full or other I/O error – clean up files and DB record
        _cleanup_files(analysis_id)
        await db.delete(analysis)
        await db.commit()
        raise HTTPException(
            status_code=507,
            detail=f"Falha ao salvar arquivo: {exc}",
        )

    # Create processing output directory
    output_dir = os.path.join(PROCESSING_DIR, analysis_id)
    os.makedirs(output_dir, exist_ok=True)

    # Start background processing in a thread
    thread = Thread(
        target=run_analysis_sync,
        args=(analysis_id, video_path, output_dir),
        daemon=True,
    )
    thread.start()

    return UploadResponse(
        id=analysis_id,
        message="Upload successful. Analysis started.",
    )


@router.get("/analyses", response_model=list[AnalysisListItem])
async def list_analyses(
    db: AsyncSession = Depends(get_db),
):
    """List all analyses, most recent first."""
    result = await db.execute(
        select(Analysis).order_by(desc(Analysis.created_at))
    )
    analyses = result.scalars().all()
    return [AnalysisListItem.model_validate(a) for a in analyses]


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full analysis results by ID."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse.model_validate(analysis)


@router.get("/analysis/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Check processing status of an analysis."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "id": analysis.id,
        "status": analysis.status,
        "progress": analysis.progress,
        "status_text": analysis.status_text,
        "error_message": analysis.error_message,
    }


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete an analysis and its associated files."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Delete files
    video_dir = os.path.join(UPLOAD_DIR, analysis_id)
    output_dir = os.path.join(PROCESSING_DIR, analysis_id)

    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    await db.delete(analysis)
    await db.commit()

    return {"message": "Analysis deleted successfully"}


@router.get("/demo-analysis", response_model=AnalysisResponse)
async def get_demo_analysis():
    """
    Return a demo analysis with realistic data for UI development and testing.
    No video upload required.
    """
    import numpy as np

    demo_id = "demo-001"

    # Generate realistic heatmap data (concentrated at center/head level)
    heatmap_points = []
    for _ in range(200):
        # Cluster around center with some spread
        x = int(np.clip(np.random.normal(960, 150), 0, 1920))
        y = int(np.clip(np.random.normal(400, 100), 0, 1080))
        heatmap_points.append({"x": x, "y": y, "value": float(np.random.uniform(1, 10))})

    # Generate timeline events
    timeline_events = []
    for t in range(0, 300, 8):
        event_types = ["shot", "shot", "shot", "combat", "ability", "kill", "death"]
        etype = np.random.choice(event_types)
        timeline_events.append({
            "timestamp": float(t + np.random.uniform(0, 5)),
            "event_type": etype,
            "description": f"{etype.capitalize()} at {t}s",
        })

    # Generate round analysis
    rounds = []
    for i in range(1, 14):
        rounds.append({
            "round_number": i,
            "timestamp_start": (i - 1) * 95.0,
            "timestamp_end": i * 95.0,
            "outcome": np.random.choice(["win", "loss"], p=[0.55, 0.45]),
            "crosshair_score": round(float(np.random.uniform(45, 85)), 1),
            "movement_score": round(float(np.random.uniform(40, 80)), 1),
            "decision_score": round(float(np.random.uniform(35, 75)), 1),
            "key_moments": [f"Key play at round {i}"],
            "notes": f"Round {i} analysis",
        })

    # Crosshair frame data for charts
    crosshair_frame_data = []
    for t in range(0, 300, 2):
        crosshair_frame_data.append({
            "timestamp": float(t),
            "head_level": bool(np.random.random() > 0.35),
            "floor_aiming": bool(np.random.random() > 0.85),
            "edge_aiming": bool(np.random.random() > 0.45),
            "combat": bool(np.random.random() > 0.8),
            "adjustment": round(float(np.random.uniform(0, 15)), 1),
        })

    # Movement frame data
    movement_frame_data = []
    for t in range(0, 300, 2):
        movement_frame_data.append({
            "timestamp": float(t),
            "moving": bool(np.random.random() > 0.4),
            "shooting": bool(np.random.random() > 0.85),
            "magnitude": round(float(np.random.uniform(0, 8)), 2),
            "peek": np.random.choice(["none", "tight", "wide", "over"], p=[0.5, 0.25, 0.15, 0.1]),
            "counter_strafe": bool(np.random.random() > 0.4),
        })

    return AnalysisResponse(
        id=demo_id,
        filename="demo_gameplay.mp4",
        status="completed",
        progress=100,
        status_text="Análise completa!",
        duration_seconds=1235.0,
        resolution="1920x1080",
        fps=60.0,
        total_frames_analyzed=6175,
        overall_score=62.4,
        crosshair_score=58.7,
        movement_score=65.2,
        decision_score=55.8,
        communication_score=72.0,
        map_score=54.3,
        crosshair_data={
            "head_level_consistency": 64.3,
            "avg_pre_aim_distance": 8.2,
            "first_contact_efficiency": 12.5,
            "center_vs_edge_ratio": 47.8,
            "floor_aiming_percentage": 18.6,
            "heatmap_points": heatmap_points,
            "frame_data": crosshair_frame_data,
        },
        movement_data={
            "counter_strafe_accuracy": 58.4,
            "movement_while_shooting": 32.1,
            "peek_type_distribution": {"tight": 45.2, "wide": 32.5, "over": 22.3},
            "spray_control_score": 68.0,
            "frame_data": movement_frame_data,
        },
        decision_data={
            "multi_angle_exposure_count": 23,
            "trade_efficiency": 42.5,
            "utility_impact_score": 55.0,
            "commitment_clarity": 61.3,
            "exposure_timeline": [
                {"timestamp": float(t), "angles": int(np.random.choice([1, 1, 2, 2, 3])), "cover": bool(np.random.random() > 0.4)}
                for t in range(0, 300, 5)
            ],
            "utility_events": [
                {"timestamp": 25.0, "type": "smoke"},
                {"timestamp": 67.0, "type": "flash"},
                {"timestamp": 112.0, "type": "molly"},
                {"timestamp": 180.0, "type": "smoke"},
                {"timestamp": 245.0, "type": "flash"},
            ],
        },
        communication_data={
            "total_callouts": 12,
            "timely_callouts_pct": 72.0,
            "late_callouts_pct": 28.0,
            "transcription_segments": [
                {"start": 15.0, "end": 17.5, "text": "Dois no A curto, um machucado", "is_callout": True, "is_timely": True},
                {"start": 45.0, "end": 46.8, "text": "Flash saindo, empurrando", "is_callout": True, "is_timely": True},
                {"start": 78.0, "end": 80.0, "text": "Rota pro B, spike avistada", "is_callout": True, "is_timely": True},
                {"start": 120.0, "end": 122.0, "text": "Boa troca, segura o site", "is_callout": True, "is_timely": False},
                {"start": 155.0, "end": 157.5, "text": "Salva, salva, salva", "is_callout": True, "is_timely": True},
                {"start": 200.0, "end": 202.0, "text": "Um no heaven, HP baixo", "is_callout": True, "is_timely": True},
            ],
            "audio_events": [],
        },
        map_data={
            "positioning_score": 54.3,
            "time_in_zones": {"a_site": 28.5, "b_site": 18.2, "mid": 22.0, "spawn": 21.3, "unknown": 10.0},
            "rotation_count": 14,
            "avg_rotation_time": 4.8,
            "exposed_positioning_pct": 38.5,
            "zone_timeline": [
                {"timestamp": 0.0, "zone": "spawn", "duration": 8.0},
                {"timestamp": 8.0, "zone": "mid", "duration": 15.0},
                {"timestamp": 23.0, "zone": "a_site", "duration": 22.0},
                {"timestamp": 45.0, "zone": "mid", "duration": 10.0},
                {"timestamp": 55.0, "zone": "b_site", "duration": 18.0},
                {"timestamp": 73.0, "zone": "spawn", "duration": 12.0},
                {"timestamp": 85.0, "zone": "a_site", "duration": 25.0},
                {"timestamp": 110.0, "zone": "mid", "duration": 8.0},
                {"timestamp": 118.0, "zone": "b_site", "duration": 20.0},
                {"timestamp": 138.0, "zone": "spawn", "duration": 6.0},
                {"timestamp": 144.0, "zone": "a_site", "duration": 30.0},
                {"timestamp": 174.0, "zone": "mid", "duration": 12.0},
                {"timestamp": 186.0, "zone": "b_site", "duration": 14.0},
            ],
            "positioning_events": [
                {"timestamp": 12.0, "event_type": "exposed", "description": "Posição exposta no mid sem cobertura", "severity": "high"},
                {"timestamp": 35.0, "event_type": "good_position", "description": "Boa ancoragem no A site", "severity": "low"},
                {"timestamp": 60.0, "event_type": "slow_rotation", "description": "Rotação lenta do mid para B (6.2s)", "severity": "medium"},
                {"timestamp": 90.0, "event_type": "exposed", "description": "Posição agressiva sem suporte do time", "severity": "high"},
                {"timestamp": 130.0, "event_type": "good_position", "description": "Bom posicionamento defensivo no B site", "severity": "low"},
                {"timestamp": 150.0, "event_type": "over_rotation", "description": "Rotação desnecessária - voltou ao A sem informação", "severity": "medium"},
            ],
        },
        timeline_events=sorted(timeline_events, key=lambda x: x["timestamp"]),
        recommendations=[
            {
                "priority": 1,
                "category": "crosshair",
                "title": "Pare de Mirar no Chão Durante Rotações",
                "description": "Você está mirando no chão 18.6% do tempo. Este é um dos hábitos mais comuns que separa jogadores de rank baixo dos de rank alto. Ao rotacionar ou se mover entre posições, sua mira SEMPRE deve estar na altura da cabeça, pronta para um possível contato.",
                "practice_drill": "Pratique rotacionar entre sites mantendo a mira na altura da cabeça. Use a técnica do 'ponto no monitor' - coloque um pequeno pedaço de fita no ponto de referência da altura da cabeça e treine para manter a mira ali.",
            },
            {
                "priority": 1,
                "category": "movement",
                "title": "Pare de Se Mover Enquanto Atira",
                "description": "Você está se movendo enquanto atira 32.1% do tempo. Valorant pune severamente a precisão em movimento. Você PRECISA estar parado ou fazendo counter-strafe quando atirar. Isso é inegociável para melhorar.",
                "practice_drill": "No Range, pratique: strafe esquerda → counter-strafe (aperte D) → atire → strafe direita → counter-strafe (aperte A) → atire. Comece devagar, aumente a velocidade. 10 minutos diários até virar memória muscular.",
            },
            {
                "priority": 2,
                "category": "crosshair",
                "title": "Mire nas Bordas, Não no Centro das Aberturas",
                "description": "Você está mirando nas bordas apenas 47.8% do tempo. Ao segurar ângulos ou picar, sua mira deve estar na BORDA onde os inimigos aparecem primeiro, não no centro de portas ou corredores.",
                "practice_drill": "Em um jogo custom, pratique segurando ângulos comuns. Posicione sua mira no 'pixel de primeiro contato' - o ponto exato onde a cabeça do inimigo será visível primeiro.",
            },
            {
                "priority": 2,
                "category": "decision",
                "title": "Reduza a Exposição a Múltiplos Ângulos",
                "description": "Você se expos a múltiplos ângulos 23 vezes. Nunca tente lutar contra múltiplos inimigos de ângulos diferentes simultaneamente. Use utilitários ou se reposicione para isolar duelos 1v1.",
                "practice_drill": "Antes de picar qualquer posição, pergunte: 'Quantos ângulos podem me ver aqui?' Se mais de 1, use smoke/flash ou encontre um ângulo melhor. Assista VODs do nAts para ver como ele isola duelos.",
            },
            {
                "priority": 3,
                "category": "movement",
                "title": "Pratique Counter-Strafing",
                "description": "Sua precisão de counter-strafe é apenas 58.4%. Counter-strafe significa pressionar a tecla de movimento oposta para parar instantaneamente antes de atirar. Esta é uma mecânica fundamental.",
                "practice_drill": "Use o Aim Lab ou o Range do Valorant. Pratique o ritmo A-D-atirar. Foque em ouvir seus tiros acertando com precisão no primeiro tiro.",
            },
        ],
        heatmap_data={
            "width": 1920,
            "height": 1080,
            "points": heatmap_points,
            "max_value": 10.0,
        },
        round_analysis=rounds,
        pro_comparison={
            "player_overall": 62.4,
            "player_crosshair": 58.7,
            "player_movement": 65.2,
            "player_decision": 55.8,
            "player_communication": 72.0,
            "player_map": 54.3,
            "benchmarks": {
                "nAts": {"crosshair": 92, "movement": 88, "decision": 95, "communication": 90, "map": 94, "overall": 92},
                "S0m": {"crosshair": 88, "movement": 85, "decision": 93, "communication": 88, "map": 90, "overall": 89},
                "TenZ": {"crosshair": 95, "movement": 92, "decision": 82, "communication": 78, "map": 80, "overall": 88},
            },
        },
    )
