"""
api/routes/history.py
=====================
Processing history and audit events endpoint.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.history import ProcessingHistoryResponse, ProcessingHistoryItem
from cloudremoval.database.repositories import ProcessingHistoryRepository

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=ProcessingHistoryResponse, summary="Get application processing history and audit log")
def get_history(
    entity_type: Optional[str] = Query(None, description="Filter by entity type: scene, inference_job, download, etc."),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
    db: Session = Depends(get_db),
) -> ProcessingHistoryResponse:
    """Retrieve immutable audit events for scenes, inference executions, and downloads."""
    events_orm = ProcessingHistoryRepository.list(db=db, entity_type=entity_type, limit=limit)
    total_count = ProcessingHistoryRepository.count(db=db)

    items = [
        ProcessingHistoryItem(
            id=e.id,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            action=e.action,
            status=e.status,
            message=e.message,
            duration_s=e.duration_s,
            created_at=e.created_at.isoformat() if e.created_at else "",
        )
        for e in events_orm
    ]

    return ProcessingHistoryResponse(
        total_count=total_count,
        events=items,
    )
