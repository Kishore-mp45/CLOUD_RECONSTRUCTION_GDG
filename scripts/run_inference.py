"""
scripts/run_inference.py
========================
DSen2-CR Local Geospatial Inference CLI — Phase 6

Executes local geospatial inference on a pair of Sentinel-2 (optical) and
Sentinel-1 (SAR) GeoTIFFs, producing a 13-band georeferenced GeoTIFF and
a side-by-side PNG preview.

Usage
-----
python scripts/run_inference.py \
    --s2 allclear_dataset/roi83528/2022_3/s2_toa/roi83528_s2_toa_2022_3_13_median.tif \
    --s1 allclear_dataset/roi83528/2022_3/s1/roi83528_s1_2022_3_24_median.tif \
    --checkpoint checkpoints/best_model.pth \
    --output-dir outputs/inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch

from cloudremoval.inference.pipeline import GeospatialInferencePipeline

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local geospatial inference with Modified DSen2-CR (Phase 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--s2",
        type=str,
        required=True,
        help="Path to input Sentinel-2 optical GeoTIFF (13 bands)",
    )
    parser.add_argument(
        "--s1",
        type=str,
        required=True,
        help="Path to input Sentinel-1 SAR GeoTIFF (2 bands)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--norm-path",
        type=str,
        default="data/normalization/normalization.json",
        help="Path to normalization parameters JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="Destination directory for outputs",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Optional custom job identifier",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Sliding window tile size",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in pixels between adjacent tiles",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Tile batch size for GPU inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (FP16)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable Automatic Mixed Precision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    s2_path = _PROJECT_ROOT / args.s2 if not Path(args.s2).is_absolute() else Path(args.s2)
    s1_path = _PROJECT_ROOT / args.s1 if not Path(args.s1).is_absolute() else Path(args.s1)
    checkpoint_path = _PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    norm_path = _PROJECT_ROOT / args.norm_path if not Path(args.norm_path).is_absolute() else Path(args.norm_path)
    output_dir = _PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    # Verification of input paths
    if not s2_path.exists():
        print(f"[FATAL] Sentinel-2 input file not found: {s2_path}", file=sys.stderr)
        sys.exit(1)
    if not s1_path.exists():
        print(f"[FATAL] Sentinel-1 input file not found: {s1_path}", file=sys.stderr)
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"[FATAL] Model checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    if not norm_path.exists():
        print(f"[FATAL] Normalization parameters not found: {norm_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"

    amp = args.amp and (device == "cuda")

    pipeline = GeospatialInferencePipeline(
        checkpoint_path=checkpoint_path,
        norm_path=norm_path,
        device=device,
        amp_enabled=amp,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )

    try:
        pipeline.run_inference(
            s2_path=s2_path,
            s1_path=s1_path,
            output_dir=output_dir,
            job_id=args.job_id,
        )
    except Exception as exc:
        log.exception("Inference failed with exception: %s", exc)
        print(f"\n[FATAL] Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
