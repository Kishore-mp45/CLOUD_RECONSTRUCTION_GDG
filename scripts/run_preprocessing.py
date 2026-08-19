"""
scripts/run_preprocessing.py
==============================
Phase 2 - ALLClear preprocessing pipeline.

Run:
    uv run python scripts/run_preprocessing.py

This script is the ONLY entry point for Phase 2 preprocessing.
All steps are visible in the terminal.  No background processes.

Pipeline:
    metadata -> validation -> triplet expansion -> scene-level split
    -> normalization stats -> patch generation -> quality filter
    -> patch sampling (if configured) -> write manifests -> report

This script NEVER modifies original ALLClear data files.
All output goes to data/manifests/ and data/normalization/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is on the path when run directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from cloudremoval.utils.logging import setup_logging
from cloudremoval.config.settings import get_settings

settings = get_settings()
setup_logging(settings)

import logging
log = logging.getLogger(__name__)


def _print(msg: str) -> None:
    """Emit a prefixed terminal line and also send it to the logger."""
    line = f"[PHASE 2] {msg}"
    print(line, flush=True)
    log.info(msg)


def _hr() -> None:
    print("=" * 60, flush=True)


def main() -> None:
    _hr()
    print("[PHASE 2] ALLClear Preprocessing Pipeline", flush=True)
    print(f"[PHASE 2] Dataset root: {settings.DATASET_ROOT}", flush=True)
    _hr()

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    dataset_root   = settings.DATASET_ROOT
    allclear_dir   = dataset_root / "allclear_dataset"
    meta_path      = dataset_root / "allclear_test_metadata.json"
    manifests_dir  = dataset_root / "data" / "manifests"
    norm_dir       = dataset_root / "data" / "normalization"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessing config from YAML
    cfg = {
        "scratch_prefix":       "/scratch/allclear/dataset_v3/dataset_30k_v4/",
        "patch_size":           256,
        "stride":               128,
        "max_nodata_frac":      0.05,
        "train_ratio":          0.80,
        "val_ratio":            0.10,
        "test_ratio":           0.10,
        "seed":                 42,
        "norm_sample_size":     500,
        "norm_version":         "v1",
        "max_patches_per_roi":  None,
        "use_all_s2_inputs":    True,
    }

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    _print("Loading metadata...")
    from cloudremoval.data.metadata import parse_metadata
    all_records = parse_metadata(meta_path, allclear_dir)
    _print(f"Total metadata records: {len(all_records)}")

    # ------------------------------------------------------------------
    # 2. Validate pairs
    # ------------------------------------------------------------------
    _print("Validating pairs (file existence, band counts, nodata)...")
    from cloudremoval.data.validation import filter_records
    valid_records, rejected = filter_records(all_records, require_s1=True)

    _print(f"Valid pairs: {len(valid_records)}")
    _print(f"Rejected pairs: {len(rejected)}")

    if rejected:
        # Summarise rejection reasons
        reasons: dict[str, int] = {}
        for r in rejected:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        # Group by first clause of reason for readability
        grouped: dict[str, int] = {}
        for reason, count in reasons.items():
            key = reason.split(":")[0].strip()
            grouped[key] = grouped.get(key, 0) + count
        for reason, count in sorted(grouped.items(), key=lambda x: -x[1]):
            _print(f"  Rejection reason | {reason}: {count}")

    # Save rejection log
    rej_path = manifests_dir / "rejected_records.json"
    with open(rej_path, "w") as fh:
        json.dump(rejected, fh, indent=2)
    _print(f"Rejection log -> {rej_path}")

    # ------------------------------------------------------------------
    # 3. Expand to triplets
    # ------------------------------------------------------------------
    _print("Expanding records to (S2_input, S1, target) triplets...")
    from cloudremoval.data.triplets import expand_to_triplets
    all_triplets = expand_to_triplets(
        valid_records,
        use_all_s2_inputs=cfg["use_all_s2_inputs"],
    )
    _print(f"Total triplets (use_all_s2_inputs={cfg['use_all_s2_inputs']}): {len(all_triplets)}")

    # ------------------------------------------------------------------
    # 4. Scene-level split (by ROI)
    # ------------------------------------------------------------------
    _print("Scene-level splitting (by ROI)...")
    from cloudremoval.data.splits import scene_level_split, verify_no_leakage
    train_t, val_t, test_t = scene_level_split(
        all_triplets,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        seed=cfg["seed"],
    )
    _print(f"Scene split - train: {len(train_t)} | val: {len(val_t)} | test: {len(test_t)}")

    unique_train_rois = len({t.roi_id for t in train_t})
    unique_val_rois   = len({t.roi_id for t in val_t})
    unique_test_rois  = len({t.roi_id for t in test_t})
    _print(f"ROI split  - train: {unique_train_rois} | val: {unique_val_rois} | test: {unique_test_rois}")

    # ------------------------------------------------------------------
    # 5. Leakage check
    # ------------------------------------------------------------------
    verify_no_leakage(train_t, val_t, test_t)
    _print("Leakage check: PASS")

    # ------------------------------------------------------------------
    # 6. Write scene-level pairs manifests
    # ------------------------------------------------------------------
    _print("Writing scene-level pair manifests...")
    from cloudremoval.data.manifests import write_pairs_manifest

    meta_payload = {
        "preprocessing_config": cfg,
        "dataset_root": str(dataset_root),
        "phase": "2",
    }
    write_pairs_manifest(all_triplets, manifests_dir / "all_pairs.json",   extra_meta=meta_payload)
    write_pairs_manifest(train_t,      manifests_dir / "train_pairs.json", extra_meta=meta_payload)
    write_pairs_manifest(val_t,        manifests_dir / "val_pairs.json",   extra_meta=meta_payload)
    write_pairs_manifest(test_t,       manifests_dir / "test_pairs.json",  extra_meta=meta_payload)

    # ------------------------------------------------------------------
    # 7. Normalization statistics (training set only)
    # ------------------------------------------------------------------
    _print(f"Computing normalization stats from {cfg['norm_sample_size']} training triplets...")
    from cloudremoval.data.normalization import (
        compute_normalization_stats, save_normalization
    )
    norm_stats = compute_normalization_stats(
        train_t,
        n_sample=cfg["norm_sample_size"],
        seed=cfg["seed"],
    )
    norm_path = norm_dir / "normalization.json"
    save_normalization(
        norm_stats, norm_path,
        version=cfg["norm_version"],
        n_sample=cfg["norm_sample_size"],
        seed=cfg["seed"],
    )
    _print(f"Normalization -> {norm_path}")
    _print(f"  S2 band means (first 4): {[f'{m:.1f}' for m in norm_stats['s2']['mean'][:4]]}")
    _print(f"  S1 band means: VV={norm_stats['s1']['mean'][0]:.2f}, VH={norm_stats['s1']['mean'][1]:.2f}")

    # ------------------------------------------------------------------
    # 8. Patch generation (train / val / test)
    # ------------------------------------------------------------------
    from cloudremoval.data.patches import generate_all_patches, sample_patches_by_roi
    from cloudremoval.data.manifests import write_patch_manifest

    patch_meta = {**meta_payload, "norm_version": cfg["norm_version"]}

    total_valid_patches = 0
    total_rejected_patches = 0

    split_results = {}
    for split_name, triplets in [("train", train_t), ("val", val_t), ("test", test_t)]:
        _print(f"Generating {split_name} patches ({len(triplets)} triplets)...")
        patches, pstats = generate_all_patches(
            triplets,
            patch_size=cfg["patch_size"],
            stride=cfg["stride"],
            max_nodata_frac=cfg["max_nodata_frac"],
            norm_version=cfg["norm_version"],
            verbose_every=200,
        )
        _print(f"  {split_name} patches valid: {len(patches)} | rejected: {pstats['patches_rejected']}")
        total_valid_patches   += len(patches)
        total_rejected_patches += pstats["patches_rejected"]

        # Optional sampling
        if cfg["max_patches_per_roi"]:
            _print(f"  Patch sampling: enabled (max {cfg['max_patches_per_roi']} per ROI)")
            patches = sample_patches_by_roi(patches, cfg["max_patches_per_roi"], seed=cfg["seed"])
            _print(f"  After sampling: {len(patches)}")
        else:
            _print("  Patch sampling: disabled (keeping all valid patches)")

        split_results[split_name] = {
            "triplets": len(triplets),
            "patches_before_sample": pstats["patches_valid"],
            "patches_final": len(patches),
            "patches_rejected": pstats["patches_rejected"],
        }

        # Write patch manifest
        manifest_path = manifests_dir / f"{split_name}.json"
        write_patch_manifest(patches, manifest_path, extra_meta=patch_meta)

    # ------------------------------------------------------------------
    # 9. Final report
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    _hr()
    _print("COMPLETE")
    _hr()
    print(f"""
  Total metadata records   : {len(all_records)}
  Valid SAR-supervised      : {len(valid_records)}
  Rejected                  : {len(rejected)}

  Total triplets            : {len(all_triplets)}
  Train / Val / Test ROIs   : {unique_train_rois} / {unique_val_rois} / {unique_test_rois}
  Train / Val / Test triplet: {len(train_t)} / {len(val_t)} / {len(test_t)}

  Train patches             : {split_results['train']['patches_final']}
  Val   patches             : {split_results['val']['patches_final']}
  Test  patches             : {split_results['test']['patches_final']}
  Total valid patches       : {total_valid_patches}
  Total rejected patches    : {total_rejected_patches}

  Patch size / stride       : {cfg['patch_size']} / {cfg['stride']}
  Max nodata fraction       : {cfg['max_nodata_frac']:.0%}
  Leakage check             : PASS
  Normalization             : {norm_path}
  Manifests                 : {manifests_dir}

  Elapsed                   : {elapsed:.1f}s
""", flush=True)

    _hr()
    print("[PHASE 2] COMPLETE", flush=True)
    _hr()


if __name__ == "__main__":
    main()
