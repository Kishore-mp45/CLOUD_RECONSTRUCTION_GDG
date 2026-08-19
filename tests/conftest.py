"""
pytest configuration and shared fixtures for cloudremoval tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pytest

# ---------------------------------------------------------------------------
# Ensure the .env is NOT loaded during tests (we use defaults/overrides)
# ---------------------------------------------------------------------------
os.environ.setdefault("DATASET_ROOT", str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def settings():
    """Return a Settings instance with test-safe defaults."""
    # Clear the lru_cache so tests always get a fresh instance
    from cloudremoval.config.settings import get_settings
    get_settings.cache_clear()
    s = get_settings()
    return s


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return Path(__file__).parent.parent
