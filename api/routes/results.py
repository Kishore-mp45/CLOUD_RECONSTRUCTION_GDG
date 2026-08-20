"""
api/routes/results.py
=====================
Inference result retrieval and result imagery preview endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from api.dependencies import get_db
from api.db.models import InferenceJob, Result, Scene
from api.schemas.results import ResultResponse
from api.services.scene_service import get_or_generate_scene_preview
from cloudremoval.evaluation.visualizer import render_s2_rgb, apply_chromaticity_match, RGB_INDICES

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("/{id}", response_model=ResultResponse, summary="Get inference result by Job or Result ID")
def get_result(
    id: str,
    db: Session = Depends(get_db),
) -> ResultResponse:
    """Retrieve spatial metadata and download URLs for a completed inference job."""
    # Look up by result_id or job_id
    res = db.query(Result).filter((Result.result_id == id) | (Result.job_id == id)).first()
    if not res:
        # Check if job is still in progress or failed
        job = db.query(InferenceJob).filter(InferenceJob.job_id == id).first()
        if job:
            raise HTTPException(
                status_code=400,
                detail=f"Inference job '{id}' is currently in state '{job.status}'. Output not yet ready.",
            )
        raise HTTPException(status_code=404, detail=f"Inference result '{id}' not found.")

    geotiff_url = f"/download?result_id={res.result_id}&file_type=geotiff"
    preview_url = f"/download?result_id={res.result_id}&file_type=png"

    infer_time = res.job.inference_time_s if res.job else None

    return ResultResponse(
        result_id=res.result_id,
        job_id=res.job_id,
        scene_id=res.scene_id,
        status="completed",
        crs=res.crs,
        width=res.width,
        height=res.height,
        resolution=res.resolution,
        band_count=res.band_count,
        geotiff_download_url=geotiff_url,
        preview_png_download_url=preview_url,
        inference_time_s=infer_time,
        created_at=res.created_at.isoformat(),
    )


@router.get("/{id}/preview/{modality}", summary="Get individual result preview PNG")
def get_result_preview_image(
    id: str,
    modality: Literal["reconstructed", "cloudy", "comparison"] = "reconstructed",
    db: Session = Depends(get_db),
) -> FileResponse:
    """Return an individual PNG preview for the reconstructed image, cloudy input, or comparison figure."""
    res = db.query(Result).filter((Result.result_id == id) | (Result.job_id == id)).first()
    if not res:
        raise HTTPException(status_code=404, detail=f"Inference result '{id}' not found.")

    if modality == "comparison":
        p = Path(res.preview_png_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="Comparison preview file not found.")
        return FileResponse(p, media_type="image/png", filename=f"{res.result_id}_comparison.png")

    elif modality == "cloudy":
        p = get_or_generate_scene_preview(db=db, scene_id=res.scene_id, modality="s2")
        if not p or not p.exists():
            raise HTTPException(status_code=404, detail="Cloudy input preview not found.")
        return FileResponse(p, media_type="image/png", filename=f"{res.scene_id}_cloudy.png")

    elif modality == "reconstructed":
        # v7: applies fixed physical-scale render_s2_rgb AND Chromaticity matching
        # to remove the green hallucination bias under thick clouds.
        # Never changes the numerical reconstruction GeoTIFF.
        out_recon_png = Path(res.preview_png_path).parent / f"{res.job_id}_reconstructed_rgb_v7.png"
        if out_recon_png.exists() and out_recon_png.stat().st_size > 1000:
            return FileResponse(out_recon_png, media_type="image/png", filename=f"{res.result_id}_reconstructed.png")

        # Read reconstructed GeoTIFF and convert to RGB
        gtiff_path = Path(res.geotiff_path)
        if gtiff_path.exists():
            try:
                import rasterio
                with rasterio.open(gtiff_path) as src:
                    arr = src.read()
                
                rgb = render_s2_rgb(arr, rgb_indices=RGB_INDICES)
                
                # Apply Chromaticity Match if original image is available
                scene = db.query(Scene).filter(Scene.scene_id == res.scene_id).first()
                if scene and scene.s2_path and Path(scene.s2_path).exists():
                    with rasterio.open(scene.s2_path) as orig_src:
                        orig_arr = orig_src.read()
                    rgb = apply_chromaticity_match(rgb, orig_arr)
                    
                plt.imsave(out_recon_png, rgb)
                return FileResponse(out_recon_png, media_type="image/png", filename=f"{res.result_id}_reconstructed.png")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[API ERROR] Error in chromaticity match or PNG save: {e}")

        # Fallback to scene target or clean RGB
        p = get_or_generate_scene_preview(db=db, scene_id=res.scene_id, modality="target")
        return FileResponse(p, media_type="image/png", filename=f"{res.result_id}_reconstructed.png")

    raise HTTPException(status_code=400, detail=f"Invalid modality: '{modality}'")
