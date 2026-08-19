"""
api/main.py
===========
FastAPI application entry point for ALLClear Cloud Removal System (Phase 8).
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

from api.db.database import init_db, SessionLocal
from api.services.db_service import seed_scenes_from_manifests
from api.routes import (
    health_router,
    scenes_router,
    inference_router,
    results_router,
    metrics_router,
    models_router,
    downloads_router,
)
from cloudremoval.config import get_settings

log = logging.getLogger("cloudremoval.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for backend startup and shutdown initialization."""
    settings = get_settings()

    print("[API] Starting FastAPI...")

    # 1. Initialize SQLite Database
    init_db()
    db = SessionLocal()
    try:
        seed_count = seed_scenes_from_manifests(db)
        print(f"[API] Database ready (Seeded {seed_count} satellite scenes)")
    finally:
        db.close()

    # 2. Check Model Checkpoint
    ckpt_path = Path(settings.CHECKPOINT_DIR) / settings.BEST_CHECKPOINT_NAME
    if ckpt_path.exists():
        print(f"[API] Model/checkpoint status: READY ({ckpt_path})")
    else:
        print(f"[API] Model/checkpoint status: WARNING — Checkpoint not found at {ckpt_path}")

    # 3. Check CUDA/GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[API] CUDA/GPU status: READY ({gpu_name})")
    else:
        print("[API] CUDA/GPU status: CPU ONLY (CUDA not available)")

    print("[API] Server ready.\n")

    yield

    print("[API] Shutting down FastAPI server.")


def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title="ALLClear Satellite Cloud Removal API",
        description="Production REST API for Sentinel-2 Optical and Sentinel-1 SAR Cloud Removal with Modified DSen2-CR.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for future frontend integration (Phase 9)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Route Controllers
    app.include_router(health_router)
    app.include_router(scenes_router)
    app.include_router(inference_router)
    app.include_router(results_router)
    app.include_router(metrics_router)
    app.include_router(models_router)
    app.include_router(downloads_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
