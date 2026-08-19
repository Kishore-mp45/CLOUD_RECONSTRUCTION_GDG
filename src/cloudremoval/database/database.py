"""
src/cloudremoval/database/database.py
=====================================
Database engine, session management, and non-destructive initialization for SQLite + SQLAlchemy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Base ORM declarative class
Base = declarative_base()


def get_default_database_url() -> str:
    """Retrieve database URL from environment or fallback to project default."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    
    # Default database inside data/ directory
    db_dir = Path("data")
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "cloudremoval.db"
    return f"sqlite:///{db_path.as_posix()}"


# Enable SQLite foreign key constraint enforcement
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable SQLite foreign key support on every connection."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


def create_db_engine(db_url: str | None = None) -> Engine:
    """Create a SQLAlchemy Engine with optimal SQLite configurations."""
    url = db_url or get_default_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, echo=False)


# Default module engine & session factory
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine: Engine | None = None) -> None:
    """
    Safely initialize all database tables (non-destructive checkfirst=True)
    and perform non-destructive schema migrations for missing columns.
    Does NOT drop existing tables or delete existing records.
    """
    eng = target_engine or engine
    # Import all models to register with Base.metadata
    from cloudremoval.database import models  # noqa: F401
    Base.metadata.create_all(bind=eng)

    # Safe SQLite column migration check
    try:
        inspector = inspect(eng)
        table_names = inspector.get_table_names()
        
        if "scenes" in table_names:
            columns = [c["name"] for c in inspector.get_columns("scenes")]
            with eng.connect() as conn:
                if "external_scene_id" not in columns:
                    conn.execute(text("ALTER TABLE scenes ADD COLUMN external_scene_id VARCHAR(256)"))
                if "source_provider" not in columns:
                    conn.execute(text("ALTER TABLE scenes ADD COLUMN source_provider VARCHAR(64) DEFAULT 'ALLClear'"))
                if "bounds_json" not in columns:
                    conn.execute(text("ALTER TABLE scenes ADD COLUMN bounds_json TEXT"))
                if "updated_at" not in columns:
                    conn.execute(text("ALTER TABLE scenes ADD COLUMN updated_at DATETIME"))
                conn.commit()

        if "inference_jobs" in table_names:
            columns = [c["name"] for c in inspector.get_columns("inference_jobs")]
            with eng.connect() as conn:
                if "model_version_id" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN model_version_id INTEGER"))
                if "batch_size" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN batch_size INTEGER DEFAULT 4"))
                if "requested_at" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN requested_at DATETIME"))
                if "started_at" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN started_at DATETIME"))
                if "completed_at" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN completed_at DATETIME"))
                if "inference_duration_s" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN inference_duration_s FLOAT"))
                if "total_duration_s" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN total_duration_s FLOAT"))
                if "updated_at" not in columns:
                    conn.execute(text("ALTER TABLE inference_jobs ADD COLUMN updated_at DATETIME"))
                conn.commit()

        if "results" in table_names:
            columns = [c["name"] for c in inspector.get_columns("results")]
            with eng.connect() as conn:
                if "bounds_json" not in columns:
                    conn.execute(text("ALTER TABLE results ADD COLUMN bounds_json TEXT"))
                if "inference_time_s" not in columns:
                    conn.execute(text("ALTER TABLE results ADD COLUMN inference_time_s FLOAT DEFAULT 0.0"))
                if "total_time_s" not in columns:
                    conn.execute(text("ALTER TABLE results ADD COLUMN total_time_s FLOAT DEFAULT 0.0"))
                if "peak_vram_gb" not in columns:
                    conn.execute(text("ALTER TABLE results ADD COLUMN peak_vram_gb FLOAT"))
                conn.commit()

        if "metric_records" in table_names:
            columns = [c["name"] for c in inspector.get_columns("metric_records")]
            with eng.connect() as conn:
                if "is_available" not in columns:
                    conn.execute(text("ALTER TABLE metric_records ADD COLUMN is_available BOOLEAN DEFAULT 0"))
                if "evaluation_source" not in columns:
                    conn.execute(text("ALTER TABLE metric_records ADD COLUMN evaluation_source VARCHAR(128) DEFAULT 'ground_truth_target'"))
                conn.commit()

    except Exception:
        pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI & service dependency generator yielding a managed database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
