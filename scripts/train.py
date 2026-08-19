"""
scripts/train.py
=================
DSen2-CR Training Script — Phase 4
===================================

DO NOT RUN THIS FILE DIRECTLY. Use the commands below.

===========================================================
TRAINING COMMANDS
===========================================================

Standard 30-epoch run (default, recommended):
    python scripts/train.py

All defaults explicit:
    python scripts/train.py --epochs 30 --batch-size 4 --amp --device cuda --seed 42

Resume from latest checkpoint (continue where you left off):
    python scripts/train.py --resume checkpoints/latest.pth

Resume and extend to 70 epochs:
    python scripts/train.py --resume checkpoints/best_model.pth --epochs 70

If CUDA OOM at batch=4 (use gradient accumulation, effective batch = 4):
    python scripts/train.py --batch-size 2 --grad-accum 2 --amp --device cuda

Disable early stopping (always train all epochs):
    python scripts/train.py --patience 0

CPU-only debugging (very slow, no AMP):
    python scripts/train.py --epochs 2 --batch-size 1 --no-amp --device cpu --num-workers 0

===========================================================
ARGUMENTS
===========================================================
  --epochs INT          Max epochs (default: 30)
  --batch-size INT      Per-GPU batch size (default: 4)
  --lr FLOAT            Learning rate (default: 1e-4)
  --weight-decay FLOAT  Weight decay (default: 1e-5)
  --device STR          cuda | cpu (default: cuda)
  --amp / --no-amp      Enable/disable AMP (default: enabled)
  --grad-accum INT      Gradient accumulation steps (default: 1)
  --num-workers INT     DataLoader workers (default: 2, Windows-safe)
  --patience INT        Early stopping patience, 0=disabled (default: 5)
  --seed INT            Random seed (default: 42)
  --resume PATH         Checkpoint path to resume from
  --checkpoint-dir PATH Checkpoint output dir (default: checkpoints)
  --log-dir PATH        Training log root (default: logs/training)
  --manifest-dir PATH   Manifests directory (default: data/manifests)
  --norm-path PATH      Normalization JSON (default: data/normalization/normalization.json)
  --base-features INT   DSen2-CR feature channels (default: 256)
  --num-res-blocks INT  DSen2-CR residual blocks (default: 16)

===========================================================
NOTES
===========================================================
  - Training runs in the foreground. The terminal MUST remain open.
  - Checkpoints are saved after every epoch.
  - best_model.pth is only overwritten when val_loss improves.
  - To stop training safely: Ctrl+C — latest.pth allows you to resume.
  - Logs are written to logs/training/YYYYMMDD_HHMMSS/
  - Early stopping stops before epoch 30 if no improvement for 5 epochs.
  - Extend to 70 epochs: --resume checkpoints/best_model.pth --epochs 70
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure src/ is on PYTHONPATH when script is run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

from cloudremoval.models import (
    DSen2CRConfig,
    build_model,
    build_loss,
)
from cloudremoval.training import (
    Trainer,
    EarlyStopping,
    TrainingLogger,
    run_preflight,
    PreflightError,
)
from cloudremoval.training.preflight import PreflightError
from cloudremoval.data.dataset import build_dataset

log = logging.getLogger(__name__)


# ===========================================================
# Argument parsing
# ===========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DSen2-CR — Phase 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python scripts/train.py --epochs 30 --batch-size 4",
    )
    # Epochs / training
    parser.add_argument("--epochs",         type=int,   default=30,   help="Max training epochs")
    parser.add_argument("--batch-size",     type=int,   default=4,    help="Batch size per GPU")
    parser.add_argument("--lr",             type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay",   type=float, default=1e-5, help="Optimizer weight decay")
    parser.add_argument("--patience",       type=int,   default=5,    help="Early stopping patience (0=disabled)")
    parser.add_argument("--seed",           type=int,   default=42,   help="Random seed")
    # Hardware
    parser.add_argument("--device",         type=str,   default="cuda", help="cuda | cpu")
    parser.add_argument("--amp",            action="store_true",  default=True,  help="Enable AMP")
    parser.add_argument("--no-amp",         action="store_false", dest="amp",    help="Disable AMP")
    parser.add_argument("--grad-accum",     type=int,   default=1,    help="Gradient accumulation steps")
    parser.add_argument("--num-workers",    type=int,   default=2,    help="DataLoader workers (2 recommended on Windows)")
    # Paths
    parser.add_argument("--resume",         type=str,   default=None, help="Resume from checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints",                         help="Checkpoint output dir")
    parser.add_argument("--log-dir",        type=str,   default="logs/training",                        help="Training log root dir")
    parser.add_argument("--manifest-dir",   type=str,   default="data/manifests",                       help="Manifests directory")
    parser.add_argument("--norm-path",      type=str,   default="data/normalization/normalization.json", help="Normalization JSON")
    # Model arch (don't change for Phase 4)
    parser.add_argument("--base-features",  type=int,   default=256,  help="DSen2-CR base feature channels")
    parser.add_argument("--num-res-blocks", type=int,   default=16,   help="Number of residual blocks")
    return parser.parse_args()


# ===========================================================
# Seed
# ===========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic mode (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    log.info("Seed set to %d", seed)


# ===========================================================
# Post-training verification
# ===========================================================

def verify_post_training(
    checkpoint_dir: Path,
    config: DSen2CRConfig,
    device: str,
    amp_enabled: bool,
) -> bool:
    """Load best_model.pth and run a forward pass to verify training succeeded.

    Returns True if all checks pass, False otherwise.
    """
    print(f"\n{'='*60}")
    print("  POST-TRAINING VERIFICATION")
    print(f"{'='*60}")

    def pcheck(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    all_ok = True

    best_path   = checkpoint_dir / "best_model.pth"
    latest_path = checkpoint_dir / "latest.pth"

    pcheck(best_path.exists(),   f"best_model.pth exists: {best_path}")
    pcheck(latest_path.exists(), f"latest.pth exists: {latest_path}")

    if not best_path.exists():
        print("  [SKIP] Cannot verify model — best_model.pth missing")
        return False

    try:
        from cloudremoval.models import build_model, load_checkpoint
        verify_model = build_model(config).to(device)
        ckpt = load_checkpoint(best_path, verify_model, device=device)
        pcheck(True, f"best_model.pth loaded (epoch={ckpt['epoch']})")
        pcheck(ckpt["epoch"] > 0, f"Checkpoint epoch is valid ({ckpt['epoch']})")

        # Forward pass
        verify_model.eval()
        from cloudremoval.models.model_config import S2_CHANNELS, S1_CHANNELS, PATCH_SIZE
        dummy_s2 = torch.randn(1, S2_CHANNELS, PATCH_SIZE, PATCH_SIZE).to(device)
        dummy_s1 = torch.randn(1, S1_CHANNELS, PATCH_SIZE, PATCH_SIZE).to(device)
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast(device_type="cuda"):
                    out = verify_model(dummy_s2, dummy_s1)
            else:
                out = verify_model(dummy_s2, dummy_s1)

        pcheck(torch.isfinite(out).all().item(), "Forward pass output is finite")
        pcheck(out.shape[1] == S2_CHANNELS, f"Output channels correct ({out.shape[1]})")

    except Exception as exc:
        pcheck(False, f"Post-training verification error: {exc}")
        all_ok = False

    print(f"{'='*60}")
    return all_ok


# ===========================================================
# Main
# ===========================================================

def main() -> None:
    args = parse_args()

    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # --- Paths ---
    manifest_dir   = _PROJECT_ROOT / args.manifest_dir
    norm_path      = _PROJECT_ROOT / args.norm_path
    checkpoint_dir = _PROJECT_ROOT / args.checkpoint_dir
    log_root       = _PROJECT_ROOT / args.log_dir

    train_manifest = manifest_dir / "train.json"
    val_manifest   = manifest_dir / "val.json"

    for p in [train_manifest, val_manifest, norm_path]:
        if not p.exists():
            print(f"[FATAL] Required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # --- Validate CUDA ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[FATAL] --device cuda requested but CUDA is NOT available.", file=sys.stderr)
        print("        Do NOT silently switch to CPU. Check CUDA installation.")
        sys.exit(1)

    amp = args.amp and device == "cuda"

    # --- Seed ---
    set_seed(args.seed)

    # --- Model config ---
    config = DSen2CRConfig(
        base_features   = args.base_features,
        num_res_blocks  = args.num_res_blocks,
        device          = device,
        amp_enabled     = amp,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        weight_decay    = args.weight_decay,
        max_epochs      = args.epochs,
        grad_accum_steps= args.grad_accum,
        val_freq        = 1,
        save_every_n_epochs = 1,
        num_workers     = args.num_workers,
    )
    config.validate()

    # --- Reproducibility snapshot ---
    repro_info = {
        "seed":              args.seed,
        "epochs":            args.epochs,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "weight_decay":      args.weight_decay,
        "amp":               amp,
        "grad_accum":        args.grad_accum,
        "device":            device,
        "patience":          args.patience,
        "train_manifest":    str(train_manifest),
        "val_manifest":      str(val_manifest),
        "norm_path":         str(norm_path),
        "base_features":     args.base_features,
        "num_res_blocks":    args.num_res_blocks,
        "torch_version":     torch.__version__,
        "cuda_version":      torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu":               torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timestamp":         datetime.now().isoformat(),
        "resume":            args.resume or "None",
    }

    # --- Training logger ---
    tlogger = TrainingLogger(log_root)
    tlogger.start(repro_info)

    # --- Datasets ---
    print("\n[TRAIN] Loading datasets...")
    train_dataset = build_dataset(train_manifest, norm_path, augment=True)
    val_dataset   = build_dataset(val_manifest,   norm_path, augment=False)
    print(f"  Train patches : {len(train_dataset):,}")
    print(f"  Val patches   : {len(val_dataset):,}")

    # Windows-safe DataLoader: persistent_workers only if num_workers > 0
    _persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size      = args.batch_size,
        shuffle         = True,
        num_workers     = args.num_workers,
        pin_memory      = (device == "cuda"),
        drop_last       = True,
        persistent_workers = _persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size      = args.batch_size,
        shuffle         = False,
        num_workers     = args.num_workers,
        pin_memory      = (device == "cuda"),
        drop_last       = False,
        persistent_workers = _persistent,
    )

    # --- Model ---
    print("\n[TRAIN] Building DSen2-CR model...")
    model = build_model(config).to(device)
    print(model.summary())

    # --- Loss ---
    criterion = build_loss(config).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    print(f"  Optimizer     : Adam (lr={args.lr}, wd={args.weight_decay})")

    # --- Scheduler: CosineAnnealingLR over full max_epochs ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = 1e-6,
    )
    print(f"  Scheduler     : CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")

    # --- Early stopping ---
    early_stopping: EarlyStopping | None = None
    if args.patience > 0:
        early_stopping = EarlyStopping(patience=args.patience, mode="min")
        print(f"  Early stopping: enabled (patience={args.patience})")
    else:
        print("  Early stopping: disabled")

    # --- Resume path ---
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = _PROJECT_ROOT / args.resume
        if not resume_path.exists():
            print(f"[FATAL] Resume checkpoint not found: {resume_path}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # PRE-FLIGHT CHECKS
    # ------------------------------------------------------------------
    try:
        preflight_info = run_preflight(
            model           = model,
            train_loader    = train_loader,
            val_loader      = val_loader,
            criterion       = criterion,
            device          = device,
            amp_enabled     = amp,
            checkpoint_dir  = checkpoint_dir,
            train_manifest  = train_manifest,
            val_manifest    = val_manifest,
            norm_path       = norm_path,
        )
    except PreflightError as e:
        print(f"\n[FATAL] Pre-flight failed: {e}", file=sys.stderr)
        print("        Training will NOT start. Fix the issue above and retry.")
        sys.exit(1)

    # Reset peak memory stats after preflight
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    trainer = Trainer(
        model           = model,
        criterion       = criterion,
        train_loader    = train_loader,
        val_loader      = val_loader,
        optimizer       = optimizer,
        config          = config,
        checkpoint_dir  = checkpoint_dir,
        early_stopping  = early_stopping,
        logger          = tlogger,
        resume_path     = resume_path,
        max_epochs      = args.epochs,
        scheduler       = scheduler,
    )

    try:
        result = trainer.train()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user (Ctrl+C).")
        print(f"  latest.pth saved at: {checkpoint_dir / 'latest.pth'}")
        print("  Resume with: python scripts/train.py --resume checkpoints/latest.pth")
        tlogger.close()
        sys.exit(0)
    finally:
        tlogger.close()

    # ------------------------------------------------------------------
    # POST-TRAINING VERIFICATION
    # ------------------------------------------------------------------
    verify_ok = verify_post_training(
        checkpoint_dir = checkpoint_dir,
        config         = config,
        device         = device,
        amp_enabled    = amp,
    )

    # ------------------------------------------------------------------
    # WRITE PHASE 4 REPORT
    # ------------------------------------------------------------------
    _write_report(
        result         = result,
        args           = args,
        config         = config,
        amp            = amp,
        train_dataset  = train_dataset,
        val_dataset    = val_dataset,
        checkpoint_dir = checkpoint_dir,
        log_run_dir    = tlogger.run_dir,
        preflight_info = preflight_info,
        verify_ok      = verify_ok,
    )

    if not verify_ok:
        print("\n[WARN] Post-training verification had issues. Check logs.")


def _write_report(
    result, args, config, amp,
    train_dataset, val_dataset,
    checkpoint_dir, log_run_dir,
    preflight_info, verify_ok,
) -> None:
    """Write docs/PHASE4_TRAINING_REPORT.md after training."""
    docs_dir = _PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    report_path = docs_dir / "PHASE4_TRAINING_REPORT.md"

    gpu_name  = preflight_info.get("gpu_name", "N/A")
    vram_gb   = preflight_info.get("vram_gb", 0.0)
    n_train   = preflight_info.get("n_train", len(train_dataset))
    n_val     = preflight_info.get("n_val",   len(val_dataset))
    n_params  = preflight_info.get("n_params", 0)

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated(0) / 1024 ** 3

    lines = [
        "# PHASE 4 TRAINING REPORT",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## 1. Dataset",
        "",
        f"| Split | Patches |",
        f"|-------|---------|",
        f"| Train | {n_train:,} |",
        f"| Val   | {n_val:,} |",
        "",
        f"- Train manifest: `data/manifests/train.json`",
        f"- Val manifest:   `data/manifests/val.json`",
        f"- Normalization:  `data/normalization/normalization.json`",
        "",
        "---",
        "",
        "## 2. Model Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Architecture | Modified DSen2-CR (early SAR concat) |",
        f"| S2 channels | 13 |",
        f"| S1 channels | 2 |",
        f"| Target channels | 13 |",
        f"| Patch size | 256×256 |",
        f"| Base features | {args.base_features} |",
        f"| Residual blocks | {args.num_res_blocks} |",
        f"| Parameters | {n_params:,} |",
        "",
        "---",
        "",
        "## 3. Training Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Optimizer | Adam |",
        f"| Learning rate | {args.lr} |",
        f"| Weight decay | {args.weight_decay} |",
        f"| Scheduler | CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6) |",
        f"| Loss | L1 (CloudRemovalLoss) |",
        f"| Batch size | {args.batch_size} |",
        f"| Grad accumulation | {args.grad_accum} |",
        f"| Effective batch | {args.batch_size * args.grad_accum} |",
        f"| AMP | {'Enabled' if amp else 'Disabled'} |",
        f"| Max epochs | {args.epochs} |",
        f"| Early stopping | patience={args.patience} |",
        f"| Seed | {args.seed} |",
        "",
        "---",
        "",
        "## 4. Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Epochs completed | {result['epochs_completed']} / {result['epochs_requested']} |",
        f"| Early stopping | {'YES' if result['early_stopped'] else 'NO'} |",
        f"| Best epoch | {result['best_epoch']} |",
        f"| Best val loss | {result['best_val_loss']:.6f} |",
        f"| Best PSNR | {result['best_psnr']:.4f} dB |",
        f"| Best SSIM | {result['best_ssim']:.6f} |",
        f"| Best MAE | {result['best_mae']:.6f} |",
        f"| Best RMSE | {result['best_rmse']:.6f} |",
        "",
        "---",
        "",
        "## 5. Training Duration",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total time | {result['total_time_s']:.1f}s ({result['total_time_s']/60:.1f} min) |",
        f"| Avg epoch time | {result['avg_epoch_time_s']:.1f}s |",
        "",
        "---",
        "",
        "## 6. Hardware",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| GPU | {gpu_name} |",
        f"| Total VRAM | {vram_gb:.1f} GB |",
        f"| Peak VRAM (training) | {peak_vram_gb:.3f} GB |",
        f"| AMP | {'Enabled' if amp else 'Disabled'} |",
        "",
        "---",
        "",
        "## 7. Checkpoint Locations",
        "",
        f"- Best model: `{checkpoint_dir / 'best_model.pth'}`",
        f"- Latest:     `{checkpoint_dir / 'latest.pth'}`",
        f"- All epochs: `{checkpoint_dir}/epoch_NNN.pth`",
        "",
        "---",
        "",
        "## 8. Logs",
        "",
        f"- Log dir:     `{log_run_dir}`",
        f"- Metrics CSV: `{log_run_dir}/training_metrics.csv`",
        f"- Config JSON: `{log_run_dir}/training_config.json`",
        f"- Metrics JSON:`{log_run_dir}/training_metrics.json`",
        "",
        "---",
        "",
        "## 9. Resume Capability",
        "",
        "```bash",
        "# Resume from latest:",
        "python scripts/train.py --resume checkpoints/latest.pth",
        "",
        "# Extend to 70 epochs:",
        "python scripts/train.py --resume checkpoints/best_model.pth --epochs 70",
        "```",
        "",
        "---",
        "",
        "## 10. Post-Training Verification",
        "",
        f"- Verification passed: {'YES' if verify_ok else 'NO'}",
        "",
        "---",
        "",
        "## 11. Next Steps",
        "",
        "- Phase 5: Test-set evaluation using `checkpoints/best_model.pth`",
        "- DO NOT start Phase 5 automatically.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written: {report_path}")


if __name__ == "__main__":
    main()
