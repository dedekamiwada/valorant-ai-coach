from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers.analysis import router as analysis_router
from app.routers.dataset import router as dataset_router
from app.routers.knowledge import router as knowledge_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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
app.include_router(dataset_router)
app.include_router(knowledge_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
