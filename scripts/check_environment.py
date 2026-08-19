"""
scripts/check_environment.py
=============================
Developer diagnostic tool for the cloudremoval project.

Usage
-----
    uv run python scripts/check_environment.py

Reports
-------
  - Python version
  - PyTorch version + CUDA version
  - CUDA availability + actual tensor test
  - GPU name + memory
  - Project paths + existence status
  - Configuration status
  - Logging status

This script does NOT train, infer, or modify any data.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on the path when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_SEP = "=" * 60


def _section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _ok(label: str, value: object) -> None:
    print(f"  [OK]      {label:<35} {value}")


def _warn(label: str, value: object) -> None:
    print(f"  [WARN]    {label:<35} {value}")


def _fail(label: str, value: object) -> None:
    print(f"  [FAIL]    {label:<35} {value}")


def _check(condition: bool, label: str, ok_val: object, fail_val: object) -> bool:
    if condition:
        _ok(label, ok_val)
    else:
        _fail(label, fail_val)
    return condition


# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------
_section("Python")
py_ok = _check(
    sys.version_info >= (3, 12),
    "Python version",
    sys.version.split()[0],
    f"{sys.version.split()[0]} (need >=3.12)",
)

# ---------------------------------------------------------------------------
# 2. PyTorch
# ---------------------------------------------------------------------------
_section("PyTorch")
try:
    import torch

    _ok("torch version", torch.__version__)
    cuda_str = torch.version.cuda or "None"
    _ok("torch.version.cuda", cuda_str)
except ImportError as exc:
    _fail("torch import", exc)
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 3. CUDA / GPU
# ---------------------------------------------------------------------------
_section("CUDA / GPU")
cuda_ready = False
if torch is not None:
    avail = torch.cuda.is_available()
    _check(avail, "torch.cuda.is_available()", True, False)

    if avail:
        count = torch.cuda.device_count()
        _ok("device count", count)
        name = torch.cuda.get_device_name(0)
        _ok("GPU name", name)

        free_b, total_b = torch.cuda.mem_get_info(0)
        total_gb = total_b / 1024**3
        free_gb = free_b / 1024**3
        _ok("GPU total memory", f"{total_gb:.2f} GB")
        _ok("GPU free memory", f"{free_gb:.2f} GB")

        # Actual tensor computation
        try:
            a = torch.ones(512, 512, device="cuda")
            b = torch.ones(512, 512, device="cuda")
            c = a @ b
            assert float(c[0, 0]) == 512.0
            _ok("CUDA tensor test", "PASSED (512x512 matmul on GPU)")
            cuda_ready = True
        except Exception as exc:
            _fail("CUDA tensor test", f"FAILED — {exc}")
    else:
        _fail("CUDA", "CUDA unavailable — check drivers and PyTorch build")
else:
    _fail("CUDA", "torch not installed")

print()
if cuda_ready:
    print("  CUDA_STATUS: READY")
else:
    print("  CUDA_STATUS: BLOCKED")

# ---------------------------------------------------------------------------
# 4. Configuration
# ---------------------------------------------------------------------------
_section("Configuration")
cfg_ok = False
try:
    from cloudremoval.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    _ok("Settings loaded", "OK")
    _ok("DEVICE", s.DEVICE)
    _ok("MAX_EPOCHS", s.MAX_EPOCHS)
    _ok("LOG_LEVEL", s.LOG_LEVEL)
    _ok("MISSING_S1_STRATEGY", s.MISSING_S1_STRATEGY)
    cfg_ok = True
except Exception as exc:
    _fail("Settings load", exc)
    s = None

# ---------------------------------------------------------------------------
# 5. Project paths
# ---------------------------------------------------------------------------
_section("Project Paths")
if s is not None:
    paths = {
        "Project root": _PROJECT_ROOT,
        "DATASET_ROOT": s.DATASET_ROOT,
        "Metadata file": s.metadata_path,
        "ALLClear dataset": s.allclear_dataset_path,
        "Checkpoints dir": s.CHECKPOINT_DIR,
        "Outputs dir": s.OUTPUT_DIR,
        "Logs dir": s.LOG_DIR,
        "DB path": s.DB_PATH,
    }
    for label, path in paths.items():
        exists = Path(path).exists()
        if exists:
            _ok(label, path)
        else:
            _warn(label, f"{path}  [NOT FOUND — will be created at runtime]")
else:
    _fail("Paths", "Cannot check — settings failed to load")

# ---------------------------------------------------------------------------
# 6. Logging
# ---------------------------------------------------------------------------
_section("Logging")
if s is not None:
    try:
        from cloudremoval.utils import setup_logging
        s.ensure_directories()
        logger = setup_logging(s)
        logger.info("check_environment.py diagnostic run")
        _ok("Logging setup", "OK")
        _ok("Log file", s.log_file_path)
    except Exception as exc:
        _fail("Logging setup", exc)
else:
    _fail("Logging", "Cannot initialise — settings failed to load")

# ---------------------------------------------------------------------------
# 7. Key imports
# ---------------------------------------------------------------------------
_section("Key Package Imports")
import_checks = [
    ("cloudremoval", "cloudremoval"),
    ("cloudremoval.config", "cloudremoval.config"),
    ("cloudremoval.utils", "cloudremoval.utils"),
    ("cloudremoval.data", "cloudremoval.data"),
    ("cloudremoval.models", "cloudremoval.models"),
    ("numpy", "numpy"),
    ("rasterio", "rasterio"),
    ("pydantic", "pydantic"),
    ("fastapi", "fastapi"),
    ("sqlalchemy", "sqlalchemy"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
]

all_imports_ok = True
for label, module in import_checks:
    try:
        __import__(module)
        _ok(f"import {label}", "OK")
    except ImportError as exc:
        _fail(f"import {label}", f"MISSING — {exc}")
        all_imports_ok = False

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
_section("Summary")
all_ok = py_ok and cuda_ready and cfg_ok and all_imports_ok
status = "ALL CHECKS PASSED" if all_ok else "ONE OR MORE CHECKS FAILED"
print(f"  {status}")
print()
print("  Phase 1 status: COMPLETE")
print("  Training:        NOT IMPLEMENTED (Phase 2+)")
print("  FastAPI:         NOT IMPLEMENTED (Phase 3+)")
print("  Frontend:        NOT IMPLEMENTED (Phase 3+)")
print("  Earth Engine:    NOT IMPLEMENTED (Phase 4+)")
print()
print(_SEP)

sys.exit(0 if all_ok else 1)
