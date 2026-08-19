"""
api/dependencies.py
===================
FastAPI dependency injection utilities.
"""

from __future__ import annotations

from typing import Generator
from sqlalchemy.orm import Session

from api.db.database import get_db

__all__ = ["get_db"]
