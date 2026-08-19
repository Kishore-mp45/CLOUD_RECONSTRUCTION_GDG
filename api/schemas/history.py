"""
api/schemas/history.py
======================
Pydantic schemas for audit trail and processing history.
"""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel


class ProcessingHistoryItem(BaseModel):
    id: int
    entity_type: str
    entity_id: str
    action: str
    status: str
    message: Optional[str] = None
    duration_s: Optional[float] = None
    created_at: str


class ProcessingHistoryResponse(BaseModel):
    total_count: int
    events: List[ProcessingHistoryItem]
