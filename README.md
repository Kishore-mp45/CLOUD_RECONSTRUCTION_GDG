# cloudremoval

> **Sentinel-2 cloud removal using Modified DSen2-CR + Sentinel-1 SAR**

## Project Overview

This system reconstructs cloud-covered Sentinel-2 optical imagery by fusing:
- **Sentinel-2 TOA** inputs (13 bands, cloudy/partially cloudy)
- **Sentinel-1 SAR** inputs (VV + VH polarisation)

using a **Modified DSen2-CR** deep-learning model (residual CNN, no cloud-mask input required).

Training is performed once on the **ALLClear** dataset. The saved best checkpoint is used for all subsequent inference. The frontend never triggers retraining.

---

## Architecture Overview

```
Sentinel-2 optical (13 bands)  ─┐
                                 ├─→  Modified DSen2-CR  →  Reconstructed S2
Sentinel-1 SAR (VV + VH)       ─┘

         ↓
    FastAPI backend
         ↓
  HTML/CSS/JS frontend
    ├── Cloudy S2 image
    ├── SAR image
    ├── Reconstructed S2 image
    ├── Before/after slider
    ├── Metrics: PSNR, SSIM, MAE, RMSE, inference time
    ├── Image metadata
    └── Download: GeoTIFF / PNG
```

**Future live satellite data**: Google Earth Engine (`code.earthengine.google.com`)

---

## Development Phase Status

| Phase | Description                             | Status             |
|-------|-----------------------------------------|--------------------|
| 0     | Machine + Dataset Audit                 | ✅ **COMPLETE**     |
| 1     | Clean Project Foundation                | ✅ **COMPLETE**     |
| 2     | ALLClear DataLoader + DSen2-CR Model    | ⏳ Not started      |
| 3     | Training Loop + Evaluation Metrics      | ⏳ Not started      |
| 4     | FastAPI Backend + Inference Pipeline    | ⏳ Not started      |
| 5     | HTML/CSS/JS Frontend                    | ⏳ Not started      |
| 6     | Google Earth Engine Integration         | ⏳ Not started      |

> **Training has NOT been implemented yet.**

---

## Requirements

- Windows 10/11
- Python 3.12+
- NVIDIA GPU with CUDA 12.1+ (tested: RTX 4060 8GB)
- [uv](https://docs.astral.sh/uv/) package manager

---

## Setup

### 1. Clone the repository

```powershell
git clone <repo-url>
cd cloudremoval
```

### 2. Configure environment

```powershell
copy .env.example .env
# Edit .env and set DATASET_ROOT to your ALLClear dataset path
notepad .env
```

### 3. Create virtual environment

```powershell
uv venv
.venv\Scripts\activate
```

### 4. Install dependencies

```powershell
uv sync
```

> PyTorch is installed with CUDA 12.1 support automatically via the `[tool.uv.sources]` configuration in `pyproject.toml`.

### 5. Install dev dependencies (for testing)

```powershell
uv sync --extra dev
```

---

## Verify CUDA

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output:
```
2.5.1+cu121
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Environment Diagnostic

```powershell
uv run python scripts/check_environment.py
```

This prints a full report of:
- Python version
- PyTorch + CUDA status
- GPU name + memory
- Project paths
- Configuration status
- Package import status

---

## Run Tests

```powershell
uv run pytest
```

All Phase 1 tests must pass before proceeding to Phase 2.

---

## Project Structure

```
cloudremoval/
├── src/
│   └── cloudremoval/          # Main Python package
│       ├── config/            # Pydantic settings (Settings, get_settings)
│       ├── utils/             # Logging, helpers
│       ├── data/              # Dataset loaders (Phase 2+)
│       ├── models/            # DSen2-CR architecture (Phase 2+)
│       ├── training/          # Training loop (Phase 2+)
│       ├── evaluation/        # PSNR/SSIM/MAE/RMSE (Phase 2+)
│       ├── inference/         # Checkpoint-based inference (Phase 3+)
│       ├── preprocessing/     # ALLClear preprocessing (Phase 2+)
│       ├── geospatial/        # CRS, alignment utilities (Phase 2+)
│       └── cloud/             # Cloud utilities (Phase 2+)
├── api/                       # FastAPI application (Phase 3+)
├── frontend/                  # HTML/CSS/JS (Phase 3+)
├── configs/
│   └── default.yaml           # Model, training, inference config
├── scripts/
│   ├── check_environment.py   # Developer diagnostic
│   └── inspect_dataset.py     # Phase 0 dataset audit
├── tests/                     # pytest test suite
├── docs/                      # Documentation + Phase 0 audit
├── checkpoints/               # Saved model checkpoints (not committed)
├── outputs/                   # Inference outputs (not committed)
├── logs/                      # Log files (not committed)
├── data/                      # SQLite DB (not committed)
├── requirements/
│   ├── base.txt               # Runtime dependencies reference
│   └── dev.txt                # Dev dependencies reference
├── .env.example               # Environment variable template
├── .gitignore
├── pyproject.toml             # Project metadata + all dependencies
└── README.md
```

---

## Configuration

All configuration is centralised in `src/cloudremoval/config/settings.py` via **Pydantic BaseSettings**.

| Variable              | Default                                   | Description                            |
|-----------------------|-------------------------------------------|----------------------------------------|
| `DATASET_ROOT`        | `D:\allclear_test_proi1_v1`               | ALLClear dataset root directory        |
| `DEVICE`              | `cuda`                                    | PyTorch device (`cuda` or `cpu`)       |
| `MAX_EPOCHS`          | `30`                                      | **Locked** maximum training epochs     |
| `CHECKPOINT_DIR`      | `checkpoints`                             | Model checkpoint directory             |
| `OUTPUT_DIR`          | `outputs`                                 | Inference output directory             |
| `LOG_DIR`             | `logs`                                    | Application log directory              |
| `DB_PATH`             | `data/cloudremoval.db`                    | SQLite database path                   |
| `MISSING_S1_STRATEGY` | `skip`                                    | S1-absent sample strategy              |
| `LOG_LEVEL`           | `INFO`                                    | Logging verbosity                      |

See `.env.example` for all available variables.

---

## Training (Phase 2+ — Not yet implemented)

Training will run **explicitly in the foreground terminal**:

```powershell
# Future command — Phase 2+
uv run python scripts/train.py
```

Terminal will show epoch/batch progress, losses, metrics, LR, timing, checkpoint saves, and GPU status.

**Training will never run in the background.**  
**The frontend never triggers retraining.**

---

## Dataset

The ALLClear dataset is located at `DATASET_ROOT` (configured in `.env`).  
It is **never copied into this repository**.

From Phase 0 audit:
- **ROIs**: 3,698
- **S2 files**: 14,792 (13 bands, float64)
- **S1 files**: 4,608 (VV + VH, float64)
- **Missing S1**: 1,203 records
- **CRS**: varies per ROI (UTM zones)
- **Resolution**: ~10 m/pixel

---

## Google Earth Engine (Phase 6+ — Not implemented)

Future live satellite data will come from:
`code.earthengine.google.com`

Collections:
- S2: `COPERNICUS/S2_SR_HARMONIZED`
- S1: `COPERNICUS/S1_GRD`
