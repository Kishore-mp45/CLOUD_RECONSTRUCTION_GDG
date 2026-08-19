"""
scripts/evaluate.py
====================
DSen2-CR Model Evaluation Script — Phase 5

Evaluates the trained model on unseen test data, calculates quantitative
metrics, generates visual comparison figures, benchmarks inference latency,
and exports evaluation reports.

Usage
-----
# Default evaluation (uses checkpoints/best_model.pth):
    python scripts/evaluate.py

# Specify manifest explicitly (e.g. India test set):
    python scripts/evaluate.py --manifest data/manifests/india/test.json

# Global test set evaluation:
    python scripts/evaluate.py --manifest data/manifests/test.json

# Custom batch size, CPU mode or custom output directory:
    python scripts/evaluate.py --output-dir outputs/evaluation --device cuda --amp --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch

from cloudremoval.evaluation import Phase5Evaluator

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DSen2-CR on unseen test data — Phase 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to best_model.pth checkpoint",
    )
    # Manifest: check if india test set exists, default to that or global test.json
    default_manifest = "data/manifests/india/test.json" if Path("data/manifests/india/test.json").exists() else "data/manifests/test.json"
    parser.add_argument(
        "--manifest",
        type=str,
        default=default_manifest,
        help="Path to test.json manifest",
    )
    parser.add_argument(
        "--norm-path",
        type=str,
        default="data/normalization/normalization.json",
        help="Path to normalization.json",
    )
    # Outputs
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for metrics and visuals",
    )
    # Hardware & Performance
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers (Windows-safe default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    checkpoint_path = _PROJECT_ROOT / args.checkpoint
    manifest_path   = _PROJECT_ROOT / args.manifest
    norm_path       = _PROJECT_ROOT / args.norm_path
    output_dir      = _PROJECT_ROOT / args.output_dir

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("        Train the model first in Phase 4 or verify the path.")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"[ERROR] Test manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if not norm_path.exists():
        print(f"[ERROR] Normalization file not found: {norm_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU for evaluation.")
        device = "cpu"

    amp = args.amp and (device == "cuda")

    evaluator = Phase5Evaluator(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        norm_path=norm_path,
        output_dir=output_dir,
        device=device,
        amp_enabled=amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    try:
        evaluator.run()
    except Exception as exc:
        log.exception("Evaluation failed with error: %s", exc)
        print(f"\n[FATAL] Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
