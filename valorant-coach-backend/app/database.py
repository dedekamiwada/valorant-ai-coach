import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "app.db")

# Ensure data dir exists (for local dev fallback)
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Lightweight SQLite schema migrations (ALTER TABLE) for new columns.
        # create_all() does not add columns to existing tables.
        async def _add_missing(table: str, needed: list[tuple[str, str]]) -> None:
            try:
                result = await conn.execute(text(f"PRAGMA table_info({table})"))
                existing = {row[1] for row in result.fetchall()}  # type: ignore[index]
                for col_name, col_type in needed:
                    if col_name not in existing:
                        await conn.execute(
                            text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                        )
            except Exception:
                # Never block app startup due to migrations.
                pass

        await _add_missing("analyses", [
            ("map_score", "FLOAT"),
            ("status_text", "VARCHAR(255)"),
            ("map_data", "TEXT"),
        ])
        await _add_missing("datasets", [
            ("analysis_progress", "INTEGER"),
            ("analysis_status_text", "TEXT"),
        ])
