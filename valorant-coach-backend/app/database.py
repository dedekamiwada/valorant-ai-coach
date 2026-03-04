import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "app.db")

# Ensure data dir exists (for local dev fallback)
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"timeout": 30},  # busy-timeout: wait up to 30s for lock
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


async def init_db():
    # Enable WAL mode so readers don't block writers and vice-versa.
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA busy_timeout=30000")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Lightweight SQLite schema migrations (ALTER TABLE) for new columns.
        # create_all() does not add columns to existing tables.
        try:
            result = await conn.execute(text("PRAGMA table_info(analyses)"))
            existing_cols = {row[1] for row in result.fetchall()}  # type: ignore[index]

            missing: list[tuple[str, str]] = []
            if "map_score" not in existing_cols:
                missing.append(("map_score", "FLOAT"))
            if "status_text" not in existing_cols:
                missing.append(("status_text", "VARCHAR(255)"))
            if "map_data" not in existing_cols:
                missing.append(("map_data", "TEXT"))

            for col_name, col_type in missing:
                await conn.execute(text(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}"))
        except Exception:
            # Never block app startup due to migrations.
            pass
