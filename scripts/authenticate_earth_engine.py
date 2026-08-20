"""Launch the official browser-based Google Earth Engine OAuth flow.

Run after setting EARTH_ENGINE_PROJECT in .env:
    uv run python scripts/authenticate_earth_engine.py
"""
from __future__ import annotations

import ee

from cloudremoval.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.EARTH_ENGINE_PROJECT:
        raise SystemExit("Set EARTH_ENGINE_PROJECT in .env before authenticating.")
    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=settings.EARTH_ENGINE_PROJECT)
    print(f"Earth Engine authentication complete for project: {settings.EARTH_ENGINE_PROJECT}")


if __name__ == "__main__":
    main()
