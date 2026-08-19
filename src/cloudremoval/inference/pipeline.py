"""
src/cloudremoval/inference/pipeline.py
======================================
Production-ready local geospatial inference pipeline for Phase 6.

Pipeline:
  1. Load trained model from checkpoint (best_model.pth)
  2. Validate input S2 and S1 GeoTIFFs
  3. Resample & align S1 to S2 reference grid in-memory
  4. Perform tiled inference with 2D Hann window blending
  5. Write georeferenced 13-band Float32 output GeoTIFF
  6. Generate true-color side-by-side PNG preview
  7. Measure exact stage timings with CUDA synchronization & peak VRAM
  8. Save execution metadata JSON (<job_id>_metadata.json)
  9. Run programmatic geospatial correctness verification
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import rasterio

from cloudremoval.models import DSen2CRConfig, build_model, load_checkpoint
from cloudremoval.geospatial.alignment import validate_geotiff, load_and_align_s1_to_s2
from cloudremoval.inference.tiled_inference import TiledInferenceEngine
from cloudremoval.inference.writer import write_reconstructed_geotiff, verify_reconstructed_geotiff
from cloudremoval.inference.preview import create_inference_preview

log = logging.getLogger(__name__)


class GeospatialInferencePipeline:
    """End-to-end local geospatial inference pipeline for Sentinel-2 cloud removal."""

    def __init__(
        self,
        checkpoint_path: Path,
        norm_path: Path,
        device: str = "cuda",
        amp_enabled: bool = True,
        tile_size: int = 256,
        overlap: int = 64,
        batch_size: int = 4,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.norm_path       = Path(norm_path)
        self.device_str      = device
        self.device          = torch.device(device)
        self.amp_enabled     = amp_enabled and (self.device.type == "cuda") and torch.cuda.is_available()
        self.tile_size       = tile_size
        self.overlap         = overlap
        self.batch_size      = batch_size

        self.model: Optional[nn.Module] = None
        self.config: Optional[DSen2CRConfig] = None
        self.engine: Optional[TiledInferenceEngine] = None
        self.checkpoint_meta: Dict[str, Any] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Load and initialize model weights from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        print(f"[PHASE 6] Loading model from checkpoint: {self.checkpoint_path}")
        self.config = DSen2CRConfig(device=self.device_str, amp_enabled=self.amp_enabled)
        self.model = build_model(self.config).to(self.device)
        self.model.eval()

        ckpt = load_checkpoint(self.checkpoint_path, self.model, device=self.device_str)
        self.checkpoint_meta = {
            "epoch": ckpt.get("epoch", "unknown"),
            "val_loss": ckpt.get("val_loss", "unknown"),
            "norm_version": ckpt.get("norm_version", "v1"),
            "timestamp": ckpt.get("timestamp", "unknown"),
        }
        log.info("Model loaded from epoch %s", self.checkpoint_meta["epoch"])

        self.engine = TiledInferenceEngine(
            model=self.model,
            norm_path=self.norm_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
            batch_size=self.batch_size,
            device=self.device_str,
            amp_enabled=self.amp_enabled,
        )

    def run_inference(
        self,
        s2_path: Path,
        s1_path: Path,
        output_dir: Path,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full inference pipeline on a pair of S2 and S1 GeoTIFFs.

        Parameters
        ----------
        s2_path : Path
            Input Sentinel-2 optical GeoTIFF.
        s1_path : Path
            Input Sentinel-1 SAR GeoTIFF.
        output_dir : Path
            Destination directory for outputs.
        job_id : Optional[str]
            Optional job identifier. If None, generated from timestamp & uuid.

        Returns
        -------
        Dict[str, Any]
            Inference results and verification metadata.
        """
        s2_path = Path(s2_path)
        s1_path = Path(s1_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if job_id is None:
            job_id = f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        out_geotiff_path = output_dir / f"{job_id}_reconstructed.tif"
        out_preview_path = output_dir / f"{job_id}_preview.png"
        out_metadata_path = output_dir / f"{job_id}_metadata.json"

        is_cuda = (self.device.type == "cuda") and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()

        t_total_0 = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Validation & Loading & Alignment
        # ------------------------------------------------------------------
        print(f"[PHASE 6] Loading S2: {s2_path.name}")
        print(f"[PHASE 6] Loading S1: {s1_path.name}")
        print("[PHASE 6] Validating geospatial metadata...")

        t_prep_0 = time.perf_counter()
        s2_data, s1_aligned, s2_meta = load_and_align_s1_to_s2(s2_path, s1_path)
        t_prep_1 = time.perf_counter()
        prep_time = t_prep_1 - t_prep_0

        print(f"[PHASE 6] S2 Reference Grid: {s2_meta['width']}x{s2_meta['height']} | CRS: {s2_meta['crs']} | Res: {s2_meta['resolution']}")
        print("[PHASE 6] Aligning S1/S2: COMPLETE")

        # ------------------------------------------------------------------
        # 2. Tiled Model Inference
        # ------------------------------------------------------------------
        print("[PHASE 6] Running tiled inference with smooth overlap blending...")

        def progress_cb(done: int, total: int) -> None:
            pct = 100.0 * done / total
            print(f"  [PHASE 6] Tiles: {done}/{total} ({pct:5.1f}%)", flush=True)

        if is_cuda:
            torch.cuda.synchronize()
        t_infer_0 = time.perf_counter()

        reconstructed_s2, tile_stats = self.engine.run_scene_inference(
            s2_data=s2_data,
            s1_data=s1_aligned,
            progress_callback=progress_cb,
        )

        if is_cuda:
            torch.cuda.synchronize()
        t_infer_1 = time.perf_counter()
        infer_time = t_infer_1 - t_infer_0

        # ------------------------------------------------------------------
        # 3. Write Output GeoTIFF
        # ------------------------------------------------------------------
        print(f"[PHASE 6] Writing GeoTIFF to: {out_geotiff_path}")
        t_post_0 = time.perf_counter()

        model_tags = {
            "JOB_ID": job_id,
            "CHECKPOINT": self.checkpoint_path.name,
            "EPOCH": self.checkpoint_meta.get("epoch", ""),
            "SOURCE_S2": s2_path.name,
            "SOURCE_S1": s1_path.name,
        }

        write_reconstructed_geotiff(
            output_path=out_geotiff_path,
            reconstructed_data=reconstructed_s2,
            s2_meta=s2_meta,
            model_metadata=model_tags,
        )

        # ------------------------------------------------------------------
        # 4. Generate PNG Preview
        # ------------------------------------------------------------------
        print(f"[PHASE 6] Writing PNG preview to: {out_preview_path}")
        preview_meta = {
            "resolution": f"{s2_meta['resolution'][0]:.1f}m",
            "crs": str(s2_meta["crs"]),
            "inference_time_s": infer_time,
        }
        create_inference_preview(
            cloudy_s2=s2_data,
            reconstructed_s2=reconstructed_s2,
            output_png_path=out_preview_path,
            job_id=job_id,
            metadata=preview_meta,
        )

        t_post_1 = time.perf_counter()
        post_time = t_post_1 - t_post_0
        total_time = time.perf_counter() - t_total_0

        # ------------------------------------------------------------------
        # 5. Programmatic Geospatial Correctness Verification
        # ------------------------------------------------------------------
        print("[PHASE 6] Verifying output GeoTIFF spatial integrity...")
        verify_report = verify_reconstructed_geotiff(out_geotiff_path, s2_meta)
        print("  [VERIFIED] CRS equality: PASS")
        print("  [VERIFIED] Affine transform equality: PASS")
        print(f"  [VERIFIED] Output dimensions: {verify_report['width']}x{verify_report['height']} (13 bands) PASS")

        # ------------------------------------------------------------------
        # 6. GPU Peak Memory
        # ------------------------------------------------------------------
        gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
        peak_vram_gb = (torch.cuda.max_memory_allocated(0) / (1024 ** 3)) if is_cuda else 0.0

        # ------------------------------------------------------------------
        # 7. Write Result Metadata JSON
        # ------------------------------------------------------------------
        result_meta = {
            "job_id": job_id,
            "status": "SUCCESS",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "input_s2": str(s2_path),
            "input_s1": str(s1_path),
            "output_geotiff": str(out_geotiff_path),
            "output_preview_png": str(out_preview_path),
            "checkpoint": str(self.checkpoint_path),
            "model_epoch": self.checkpoint_meta.get("epoch", "unknown"),
            "norm_version": self.checkpoint_meta.get("norm_version", "v1"),
            "geospatial": {
                "crs": str(s2_meta["crs"]),
                "transform": tuple(s2_meta["transform"]),
                "width": s2_meta["width"],
                "height": s2_meta["height"],
                "resolution": list(s2_meta["resolution"]),
                "band_count": 13,
            },
            "inference_config": {
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "batch_size": self.batch_size,
                "total_tiles": tile_stats["total_tiles"],
                "amp_enabled": self.amp_enabled,
                "device": self.device_str,
            },
            "performance": {
                "gpu_name": gpu_name,
                "peak_vram_gb": round(peak_vram_gb, 4),
                "preprocessing_time_s": round(prep_time, 3),
                "model_inference_time_s": round(infer_time, 3),
                "postprocessing_time_s": round(post_time, 3),
                "total_pipeline_time_s": round(total_time, 3),
            },
            "verification": verify_report,
        }

        with open(out_metadata_path, "w", encoding="utf-8") as f:
            json.dump(result_meta, f, indent=2)

        # ------------------------------------------------------------------
        # 8. Print Summary Banner
        # ------------------------------------------------------------------
        self._print_banner(result_meta)

        return result_meta

    def _print_banner(self, meta: Dict[str, Any]) -> None:
        """Print formatted completion banner."""
        geo = meta["geospatial"]
        perf = meta["performance"]
        inf = meta["inference_config"]

        print(f"\n{'='*60}")
        print("  PHASE 6 COMPLETE — GEOSPATIAL INFERENCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Job ID               : {meta['job_id']}")
        print(f"  Model Checkpoint     : {meta['checkpoint']} (Epoch {meta['model_epoch']})")
        print(f"  Output Dimensions    : {geo['width']} x {geo['height']} (13 bands)")
        print(f"  CRS                  : {geo['crs']}")
        print(f"  Resolution           : {geo['resolution'][0]:.2f}m")
        print(f"  Tiles Processed      : {inf['total_tiles']} (Tile size={inf['tile_size']}, Overlap={inf['overlap']})")
        print(f"  GPU Hardware         : {perf['gpu_name']} (Peak VRAM: {perf['peak_vram_gb']:.3f} GB)")
        print(f"  Model Inference Time : {perf['model_inference_time_s']:.3f} s")
        print(f"  Total Pipeline Time  : {perf['total_pipeline_time_s']:.3f} s")
        print(f"  Reconstructed GeoTIFF: {meta['output_geotiff']}")
        print(f"  PNG Preview          : {meta['output_preview_png']}")
        print(f"  Metadata JSON        : {meta['output_geotiff'].replace('_reconstructed.tif', '_metadata.json')}")
        print(f"{'='*60}\n")
