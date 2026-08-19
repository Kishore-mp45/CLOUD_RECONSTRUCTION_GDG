"""
api/services/db_service.py
==========================
Database seeding and initial population from Phase 2 manifests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from sqlalchemy.orm import Session

from api.db.models import Scene
from cloudremoval.config import get_settings

log = logging.getLogger(__name__)


def seed_scenes_from_manifests(db: Session) -> int:
    """Populate database with available satellite scenes from manifests if empty."""
    existing_count = db.query(Scene).count()
    if existing_count > 0:
        log.debug("Database already contains %d scenes. Skipping seed.", existing_count)
        return existing_count

    settings = get_settings()
    manifest_paths = [
        Path("data/manifests/india/test.json"),
        Path("data/manifests/india/val.json"),
        Path("data/manifests/test.json"),
    ]

    added = 0
    seen_ids = set()

    for m_path in manifest_paths:
        if not m_path.exists():
            continue

        try:
            with open(m_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            patches = data.get("patches", [])
            for p in patches:
                scene_id = p.get("patch_id") or p.get("pair_id")
                if not scene_id or scene_id in seen_ids:
                    continue

                seen_ids.add(scene_id)
                s2_p = p.get("s2_path", "")
                s1_p = p.get("s1_path", "")
                target_p = p.get("target_path", "")
                roi_id = p.get("roi_id", "unknown_roi")
                dates = p.get("dates", {})
                acq_time = dates.get("s2_input_dates", [""])[0] if dates.get("s2_input_dates") else "2022-01-01"

                # Cloud density default: India test/val patches are cloudy samples (density >= 60%)
                # Let's seed realistic density values based on patch index for demonstration
                # (e.g. 75.0% - 95.0% for cloudy scenes, with some lower for testing filtering)
                cloud_density = 75.0 + (hash(scene_id) % 25)
                # Ensure a couple are below 60% for negative test verification
                if "clear" in scene_id.lower() or (hash(scene_id) % 7 == 0):
                    cloud_density = 25.0

                is_eligible = cloud_density >= 60.0

                scene = Scene(
                    scene_id=scene_id,
                    roi_id=roi_id,
                    acquisition_time=acq_time,
                    s2_path=str(s2_p),
                    s1_path=str(s1_p),
                    target_path=str(target_p) if target_p else None,
                    cloud_density_percent=float(cloud_density),
                    cloud_probability_threshold=60.0,
                    is_eligible=is_eligible,
                    crs="EPSG:32643",
                    width=256,
                    height=256,
                    resolution=10.0,
                )
                db.add(scene)
                added += 1

        except Exception as exc:
            log.warning("Failed to seed from manifest %s: %s", m_path, exc)

    if added > 0:
        db.commit()
        log.info("Seeded %d satellite scenes into database.", added)

    return added
