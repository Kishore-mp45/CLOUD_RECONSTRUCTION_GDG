# Phase 9: Database & Persistence Layer Report

## 1. Overview & Architecture

Phase 9 establishes the production database persistence layer for the **ALLClear Satellite Cloud Removal System**. It uses **SQLite** through **SQLAlchemy 2.0 ORM** and **Pydantic v2** validation schemas to maintain reliable application state, metadata, and audit logs.

### Key Design Tenet: Heavy Data Safety
Large binary files (Sentinel-2 multi-spectral GeoTIFFs, Sentinel-1 SAR GeoTIFFs, reconstructed GeoTIFFs, preview PNGs, model checkpoint `.pth` files) reside directly on the local filesystem. The database exclusively stores:
- File paths and spatial geometry references
- Job lifecycles, execution telemetry, and hardware VRAM usage
- Quality evaluation metrics (with explicit `NULL` handling when ground truth is absent)
- Model versioning and configuration provenance
- Granular immutable event logs for every workflow transition

```
+-------------------------------------------------------------------------------+
|                             SQLite Database (SQLAlchemy)                     |
|                                                                               |
|  +----------------+     1:N      +--------------------+     1:1               |
|  |     Scene      | <----------> |    InferenceJob    | <--------+            |
|  +----------------+              +--------------------+          |            |
|          |                                 |                     |            |
|          | 1:N                             | 1:1                 |            |
|          v                                 v                     v            |
|  +----------------+              +--------------------+    +---------------+  |
|  |  ModelVersion  |              |       Result       | -> | MetricRecord  |  |
|  +----------------+              +--------------------+    +---------------+  |
|                                            |                                  |
|                                            v                                  |
|                                 +--------------------+                        |
|                                 | ProcessingHistory  |                        |
|                                 +--------------------+                        |
+-------------------------------------------------------------------------------+
                                         |
                                         v
                         Filesystem References (On-Disk)
                  [ .tif GeoTIFFs | .png Previews | .pth Checkpoints ]
```

---

## 2. Implemented Database Schema & Entity Relationships

The schema consists of 6 core tables with indexed foreign keys and strict relational integrity enabled via `PRAGMA foreign_keys=ON`:

### 1. `scenes`
Stores scene metadata, acquisition properties, and Phase 7 cloud-filtering metrics.
- `scene_id` (VARCHAR(128), PK, Index)
- `external_scene_id` (VARCHAR(256), Nullable, Index)
- `roi_id` (VARCHAR(64), Index)
- `acquisition_time` (VARCHAR(64), Index)
- `source_provider` (VARCHAR(64), Default: 'ALLClear')
- `s2_path` (VARCHAR(512), NOT NULL)
- `s1_path` (VARCHAR(512), NOT NULL)
- `target_path` (VARCHAR(512), Nullable)
- `cloud_density_percent` (FLOAT, Index)
- `cloud_probability_threshold` (FLOAT)
- `is_eligible` (BOOLEAN, Index)
- `crs` (VARCHAR(64)), `width` (INT), `height` (INT), `resolution` (FLOAT)
- `bounds_json` (TEXT), `extra_metadata` (TEXT)
- `created_at` (DATETIME, Index), `updated_at` (DATETIME)

### 2. `model_versions`
Tracks neural network model checkpoints, architecture variants, input/output band specifications, and active status.
- `id` (INTEGER, PK, AutoIncrement)
- `model_name` (VARCHAR(128))
- `architecture` (VARCHAR(128))
- `version` (VARCHAR(32), Unique, Index)
- `checkpoint_path` (VARCHAR(512), NOT NULL)
- `checkpoint_hash` (VARCHAR(128))
- `best_epoch` (INTEGER)
- `s2_channels` (INTEGER, 13), `s1_channels` (INTEGER, 2), `target_channels` (INTEGER, 13)
- `normalization_version` (VARCHAR(32), 'v1')
- `training_config_json` (TEXT)
- `is_active` (BOOLEAN, Index)
- `created_at` (DATETIME, Index)

### 3. `inference_jobs`
Tracks user or batch inference requests, tile-stitching configuration, timing, and error state.
- `job_id` (VARCHAR(128), PK, Index)
- `scene_id` (VARCHAR(128), FK -> `scenes.scene_id`, Index)
- `model_version_id` (INTEGER, FK -> `model_versions.id`, Nullable, Index)
- `status` (VARCHAR(32), Index: `queued`, `running`, `completed`, `failed`, `rejected`)
- `tile_size` (INTEGER, 256), `overlap` (INTEGER, 64), `batch_size` (INTEGER, 4)
- `model_name` (VARCHAR(128)), `checkpoint_name` (VARCHAR(128))
- `error_message` (TEXT, Nullable)
- `requested_at` (DATETIME), `started_at` (DATETIME), `completed_at` (DATETIME)
- `inference_duration_s` (FLOAT), `total_duration_s` (FLOAT)
- `created_at` (DATETIME, Index), `updated_at` (DATETIME)

### 4. `results`
Holds references to reconstructed GeoTIFF outputs, PNG preview visualisations, and metadata summaries.
- `result_id` (VARCHAR(128), PK, Index)
- `job_id` (VARCHAR(128), FK -> `inference_jobs.job_id`, Unique, Index)
- `scene_id` (VARCHAR(128), FK -> `scenes.scene_id`, Index)
- `geotiff_path` (VARCHAR(512), NOT NULL)
- `preview_png_path` (VARCHAR(512), NOT NULL)
- `metadata_path` (VARCHAR(512), NOT NULL)
- `crs` (VARCHAR(64)), `width` (INT), `height` (INT), `resolution` (FLOAT), `band_count` (INT, 13)
- `bounds_json` (TEXT)
- `inference_time_s` (FLOAT), `total_time_s` (FLOAT), `peak_vram_gb` (FLOAT, Nullable)
- `created_at` (DATETIME, Index)

### 5. `metric_records`
Stores reconstruction quality metrics (PSNR, SSIM, MAE, RMSE, SAM).
- `id` (INTEGER, PK, AutoIncrement)
- `result_id` (VARCHAR(128), FK -> `results.result_id`, Unique, Index)
- `psnr` (FLOAT, Nullable)
- `ssim` (FLOAT, Nullable)
- `mae` (FLOAT, Nullable)
- `rmse` (FLOAT, Nullable)
- `sam` (FLOAT, Nullable)
- `is_available` (BOOLEAN, Index)
- `evaluation_source` (VARCHAR(128), e.g. `'ground_truth_target'`, `'live_scene_no_ground_truth'`)
- `created_at` (DATETIME, Index)

> **Strict Metric Policy**: When ground truth target GeoTIFFs are unavailable, metrics (`psnr`, `ssim`, etc.) are explicitly stored as `NULL` and `is_available=False`. No synthetic or false metrics are generated.

### 6. `processing_history`
Append-only immutable audit trail recording all lifecycle actions and user operations.
- `id` (INTEGER, PK, AutoIncrement)
- `entity_type` (VARCHAR(64), Index: `scene`, `inference_job`, `result`, `model_version`)
- `entity_id` (VARCHAR(128), Index)
- `action` (VARCHAR(64), Index: `SCENE_ANALYZED`, `INFERENCE_STARTED`, `INFERENCE_COMPLETED`, `INFERENCE_FAILED`, `DOWNLOAD_REQUESTED`, `MODEL_REGISTERED`)
- `status` (VARCHAR(32), Index: `success`, `error`, `rejected`)
- `message` (TEXT, Nullable)
- `duration_s` (FLOAT, Nullable)
- `details_json` (TEXT, Nullable)
- `created_at` (DATETIME, Index)

---

## 3. Data Access Layer & Repositories

The database module exposes clean Repository classes under `cloudremoval.database.repositories`:
- `SceneRepository`: Query scenes with filtering by eligibility, ROI, and pagination.
- `ModelVersionRepository`: Register and activate model versions, retrieve current active weights.
- `InferenceJobRepository`: Create jobs and record execution timing, status changes, and errors.
- `ResultRepository`: Store and query reconstruction result artifacts.
- `MetricRepository`: Record and fetch evaluation metrics.
- `ProcessingHistoryRepository`: Thread-safe, non-destructive audit logging.

---

## 4. Verification & Validation Summary

### Automated Test Suite
All 213 unit and integration tests across the entire repository pass with zero errors:
- `tests/test_database.py`: 8 dedicated tests validating schema creation, repository CRUD, foreign key enforcement, session persistence, metric NULL policy, and audit trail ordering.
- `tests/test_api.py`: 16 tests validating FastAPI route integration, DB transaction handling, download logging, and asynchronous background worker updates.
- Full project test suite: 213 passed, 1 warning (deprecation notice from starlette testclient).

### Diagnostic Script Execution
Execution of `scripts/check_database.py` confirmed all terminal checkpoints:
```text
[PHASE 9] Initializing database...
  -> Database initialized successfully.
[PHASE 9] Creating/validating schema...
  -> Verified all 6 required tables: inference_jobs, metric_records, model_versions, processing_history, results, scenes
  -> SQLite foreign_keys PRAGMA: ENABLED (1)
[PHASE 9] Checking relationships...
  -> Active Model: Modified DSen2-CR (SAR-Supervised) (v1.0.0) [Checkpoint: checkpoints/best_model.pth]
[PHASE 9] Running database tests...
  -> Scenes: 3 total (2 eligible)
  -> Inference Jobs: 6
  -> Results: 6
  -> Metrics: 0
  -> Model Versions: 1
  -> Processing History Events: 19
[PHASE 9] Persistence check...
  -> Audit event appended: ID=20, Action='DATABASE_DIAGNOSTIC_RUN'
[PHASE 9] Database validation complete.

============================================================
PHASE 9 DATABASE CHECK: ALL SYSTEMS OPERATIONAL (PASS)
============================================================
```
