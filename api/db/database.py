"""
api/db/database.py
==================
SQLAlchemy engine, session factory, and database initialization for Phase 8.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from cloudremoval.config import get_settings

log = logging.getLogger(__name__)

settings = get_settings()

# Ensure database directory exists
db_file = Path(settings.DB_PATH)
db_file.parent.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_file.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables without dropping existing data."""
    log.info("Initializing SQLite database at: %s", db_file)
    Base.metadata.create_all(bind=engine)
    log.info("Database tables initialized successfully.")
