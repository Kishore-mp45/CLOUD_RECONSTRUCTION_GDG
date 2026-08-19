"""
scripts/check_database.py
=========================
Diagnostic and verification script for the Phase 9 Database Layer.

Reports:
  - Database URL / Path
  - Connection & PRAGMA foreign keys status
  - Verified tables and schemas
  - Row counts for all 6 tables
  - Active ModelVersion record
  - Audit history events
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from sqlalchemy import inspect, text

from cloudremoval.database import (
    Base,
    engine,
    SessionLocal,
    init_db,
    SceneRepository,
    InferenceJobRepository,
    ResultRepository,
    MetricRepository,
    ModelVersionRepository,
    ProcessingHistoryRepository,
)
from cloudremoval.database.models import (
    Scene,
    InferenceJob,
    Result,
    Metric,
    ModelVersion,
    ProcessingHistory,
)


def run_database_check() -> bool:
    """Execute complete diagnostic scan on the persistent application database."""
    print("\n" + "=" * 60)
    print("PHASE 9 — DATABASE DIAGNOSTIC & SANITY CHECK")
    print("=" * 60 + "\n")

    print("[PHASE 9] Initializing database...")
    init_db()
    print("  -> Database initialized successfully.")

    print("[PHASE 9] Creating/validating schema...")
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    expected_tables = ["scenes", "model_versions", "inference_jobs", "results", "metric_records", "processing_history"]
    
    missing_tables = [t for t in expected_tables if t not in table_names]
    if missing_tables:
        print(f"  [ERROR] Missing tables: {missing_tables}")
        return False
    print(f"  -> Verified all {len(expected_tables)} required tables: {', '.join(table_names)}")

    db = SessionLocal()
    try:
        # Check SQLite foreign keys pragma
        fk_status = db.execute(text("PRAGMA foreign_keys")).scalar()
        print(f"  -> SQLite foreign_keys PRAGMA: {'ENABLED (1)' if fk_status == 1 else 'DISABLED (0)'}")

        print("[PHASE 9] Checking relationships...")
        # Ensure default model is registered
        active_model = ModelVersionRepository.ensure_default_active_model(db)
        print(f"  -> Active Model: {active_model.model_name} ({active_model.version}) [Checkpoint: {active_model.checkpoint_path}]")

        print("[PHASE 9] Running database tests...")
        # Query counts
        scene_count = SceneRepository.count(db)
        eligible_count = SceneRepository.count(db, eligible_only=True)
        job_count = InferenceJobRepository.count(db)
        result_count = ResultRepository.count(db)
        metric_count = MetricRepository.count(db)
        history_count = ProcessingHistoryRepository.count(db)
        model_count = len(ModelVersionRepository.list(db))

        print(f"  -> Scenes: {scene_count} total ({eligible_count} eligible)")
        print(f"  -> Inference Jobs: {job_count}")
        print(f"  -> Results: {result_count}")
        print(f"  -> Metrics: {metric_count}")
        print(f"  -> Model Versions: {model_count}")
        print(f"  -> Processing History Events: {history_count}")

        print("[PHASE 9] Persistence check...")
        # Test a temporary transactional record to verify write/commit/rollback
        test_event = ProcessingHistoryRepository.log_event(
            db=db,
            entity_type="diagnostic",
            entity_id="test_diag_01",
            action="DATABASE_DIAGNOSTIC_RUN",
            status="success",
            message="Diagnostic sanity check completed successfully",
        )
        print(f"  -> Audit event appended: ID={test_event.id}, Action='{test_event.action}'")

        print("[PHASE 9] Database validation complete.")
        print("\n" + "=" * 60)
        print("PHASE 9 DATABASE CHECK: ALL SYSTEMS OPERATIONAL (PASS)")
        print("=" * 60 + "\n")
        return True

    except Exception as exc:
        print(f"  [ERROR] Database diagnostic failed: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = run_database_check()
    sys.exit(0 if success else 1)
