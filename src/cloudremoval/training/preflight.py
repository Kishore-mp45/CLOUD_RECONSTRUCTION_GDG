"""
src/cloudremoval/training/preflight.py
========================================
Pre-flight checks for Phase 4 training.

Runs a series of checks BEFORE the actual training loop starts.
The preflight does NOT count as an epoch — it exists only to
catch configuration problems early and give clear error messages.

Checks:
  1.  CUDA available
  2.  RTX 4060 detected (or any CUDA GPU if different hardware)
  3.  Train manifest exists and is non-empty
  4.  Val manifest exists and is non-empty
  5.  Normalization file exists
  6.  Model loaded with correct channel counts
  7.  Checkpoint directory is writable
  8.  One training batch: forward + backward + loss finite
  9.  One validation batch: forward + loss finite
  10. AMP works (if enabled)
  11. No CUDA OOM on a single batch

All checks print [PASS] or [FAIL] to stdout.
On any [FAIL], a PreflightError is raised with a clear message.
Training must NOT start if preflight fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

_SEP = "=" * 60


class PreflightError(RuntimeError):
    """Raised when a pre-flight check fails."""


def _check(cond: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    if not cond:
        raise PreflightError(f"Preflight FAILED: {label}{suffix}")


def run_preflight(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    amp_enabled: bool,
    checkpoint_dir: Path,
    train_manifest: Path,
    val_manifest: Path,
    norm_path: Path,
) -> dict:
    """Run all pre-flight checks.

    Parameters
    ----------
    model : nn.Module
        DSen2CR model (already on device).
    train_loader, val_loader : DataLoader
    criterion : nn.Module
        Loss function (already on device).
    device : str
        "cuda" or "cpu".
    amp_enabled : bool
    checkpoint_dir : Path
    train_manifest, val_manifest, norm_path : Path

    Returns
    -------
    dict
        Summary info collected during preflight.

    Raises
    ------
    PreflightError
        On any failed check.
    """
    print(f"\n{_SEP}")
    print("  PRE-FLIGHT CHECKS")
    print(_SEP)

    info: dict = {}

    # ------------------------------------------------------------------
    # 1. CUDA
    # ------------------------------------------------------------------
    cuda_ok = torch.cuda.is_available()
    _check(cuda_ok, "CUDA is available")

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        info["gpu_name"]  = gpu_name
        info["vram_gb"]   = vram_gb
        _check(True, f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        # Warn if not RTX 4060 (don't fail — user may have different GPU)
        is_rtx4060 = "4060" in gpu_name
        if not is_rtx4060:
            print(f"  [WARN] Expected RTX 4060 — found {gpu_name}")
            print(f"         Training will continue but VRAM budget may differ.")

    # ------------------------------------------------------------------
    # 2. Files exist
    # ------------------------------------------------------------------
    _check(train_manifest.exists(), "Train manifest exists", str(train_manifest))
    _check(val_manifest.exists(),   "Val manifest exists",   str(val_manifest))
    _check(norm_path.exists(),      "Normalization file exists", str(norm_path))

    # ------------------------------------------------------------------
    # 3. Dataset sizes
    # ------------------------------------------------------------------
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    _check(n_train > 0, f"Train dataset non-empty ({n_train} patches)")
    _check(n_val   > 0, f"Val dataset non-empty ({n_val} patches)")
    info["n_train"] = n_train
    info["n_val"]   = n_val

    # ------------------------------------------------------------------
    # 4. Model channel sanity
    # ------------------------------------------------------------------
    from cloudremoval.models.model_config import S2_CHANNELS, S1_CHANNELS, TARGET_CHANNELS, PATCH_SIZE
    try:
        n_params = sum(p.numel() for p in model.parameters())
        _check(n_params > 1_000_000, f"Model has sufficient parameters ({n_params:,})")
        info["n_params"] = n_params
    except Exception as exc:
        _check(False, "Model parameter count", str(exc))

    # ------------------------------------------------------------------
    # 5. Checkpoint dir writable
    # ------------------------------------------------------------------
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        test_file = checkpoint_dir / ".preflight_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        _check(True, f"Checkpoint directory writable: {checkpoint_dir}")
    except Exception as exc:
        _check(False, "Checkpoint directory writable", str(exc))

    # ------------------------------------------------------------------
    # 6. One training batch: forward + backward + loss finite
    # ------------------------------------------------------------------
    print(f"\n  Running one-batch train test...")
    try:
        model.train()
        batch = next(iter(train_loader))
        s2     = batch["s2"].to(device, non_blocking=True)
        s1     = batch["s1"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        # Channel check
        _check(s2.shape[1]     == S2_CHANNELS,     f"S2 channels={s2.shape[1]} (expected {S2_CHANNELS})")
        _check(s1.shape[1]     == S1_CHANNELS,      f"S1 channels={s1.shape[1]} (expected {S1_CHANNELS})")
        _check(target.shape[1] == TARGET_CHANNELS,  f"Target channels={target.shape[1]} (expected {TARGET_CHANNELS})")
        _check(s2.shape[2]     == PATCH_SIZE,       f"Patch height={s2.shape[2]} (expected {PATCH_SIZE})")
        _check(s2.shape[3]     == PATCH_SIZE,       f"Patch width={s2.shape[3]} (expected {PATCH_SIZE})")

        # Finite input check
        _check(torch.isfinite(s2).all().item(),     "S2 input batch is finite")
        _check(torch.isfinite(s1).all().item(),     "S1 input batch is finite")
        _check(torch.isfinite(target).all().item(), "Target batch is finite")

        # Forward
        optimizer_test = torch.optim.SGD(model.parameters(), lr=1e-10)
        optimizer_test.zero_grad()

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(s2, s1)
                loss, _ = criterion(pred, target)
        else:
            pred = model(s2, s1)
            loss, _ = criterion(pred, target)

        _check(torch.isfinite(loss).item(), f"Train batch loss is finite (loss={loss.item():.6f})")
        _check(tuple(pred.shape) == (s2.shape[0], TARGET_CHANNELS, PATCH_SIZE, PATCH_SIZE),
               f"Model output shape correct {tuple(pred.shape)}")

        # Backward
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        _check(has_grad, "Gradients exist after backward pass")
        optimizer_test.zero_grad()

    except PreflightError:
        raise
    except torch.cuda.OutOfMemoryError as oom:
        _check(False, "No CUDA OOM on single training batch", str(oom))
    except Exception as exc:
        _check(False, "One-batch train test", str(exc))

    # ------------------------------------------------------------------
    # 7. One validation batch
    # ------------------------------------------------------------------
    print(f"\n  Running one-batch val test...")
    try:
        model.eval()
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            sv2     = val_batch["s2"].to(device, non_blocking=True)
            sv1     = val_batch["s1"].to(device, non_blocking=True)
            starget = val_batch["target"].to(device, non_blocking=True)

            if amp_enabled:
                with torch.amp.autocast(device_type="cuda"):
                    vpred = model(sv2, sv1)
                    vloss, _ = criterion(vpred, starget)
            else:
                vpred = model(sv2, sv1)
                vloss, _ = criterion(vpred, starget)

        _check(torch.isfinite(vloss).item(), f"Val batch loss is finite (loss={vloss.item():.6f})")

    except PreflightError:
        raise
    except Exception as exc:
        _check(False, "One-batch val test", str(exc))

    # ------------------------------------------------------------------
    # 8. AMP
    # ------------------------------------------------------------------
    if amp_enabled:
        print(f"\n  AMP test...")
        try:
            model.train()
            scaler = torch.amp.GradScaler(device="cuda")
            opt2   = torch.optim.SGD(model.parameters(), lr=1e-10)
            opt2.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                ap = model(s2, s1)
                al, _ = criterion(ap, target)

            scaler.scale(al).backward()
            scaler.step(opt2)
            scaler.update()
            opt2.zero_grad()

            _check(torch.isfinite(al).item(), f"AMP loss is finite (loss={al.item():.6f})")
        except PreflightError:
            raise
        except Exception as exc:
            _check(False, "AMP one-batch test", str(exc))
    else:
        print(f"  [SKIP] AMP test (AMP disabled)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print("  PRE-FLIGHT COMPLETE — All checks passed")
    print(_SEP)

    if cuda_ok:
        torch.cuda.empty_cache()

    return info
