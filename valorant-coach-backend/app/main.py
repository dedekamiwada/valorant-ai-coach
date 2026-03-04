import os
import shutil
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.database import init_db, engine
from app.routers.analysis import router as analysis_router


def _purge_legacy_uploads() -> None:
    """Remove upload/processing dirs that older versions stored on /data.

    Previous code saved videos under /data/uploads and /data/processing which
    shares the small persistent volume with the SQLite database.  This one-time
    cleanup frees that space so the DB can operate normally.
    """
    data_dir = os.environ.get("DATA_DIR", "/data")
    for subdir in ("uploads", "processing"):
        path = os.path.join(data_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _purge_legacy_uploads()
    await init_db()

    # Mark any analyses left in "processing" or "pending" as "failed".
    # This happens when the previous instance was OOM-killed or crashed.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE analyses SET status = 'failed', "
                "error_message = 'Servidor reiniciou durante a análise. Tente novamente.', "
                "updated_at = :now "
                "WHERE status IN ('processing', 'pending')"
            ),
            {"now": datetime.now(timezone.utc).isoformat()},
        )

    # Reclaim any freed space inside the SQLite file.
    # VACUUM cannot run inside a transaction, so use AUTOCOMMIT.
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.exec_driver_sql("VACUUM")
    yield


app = FastAPI(
    title="Valorant AI Coach",
    description="AI-powered VOD analysis for Valorant gameplay coaching",
    version="1.0.0",
    lifespan=lifespan,
)

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(analysis_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
