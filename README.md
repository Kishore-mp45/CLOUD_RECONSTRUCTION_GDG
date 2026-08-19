# ALLClear Satellite Cloud Removal System

> **Deep Learning Geospatial Cloud Removal using Modified DSen2-CR with Sentinel-2 Optical and Sentinel-1 SAR Radar Fusion**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%20CUDA-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/pytest-224%20passed-brightgreen.svg)]()

---

## 1. Overview

The **ALLClear Cloud Removal System** reconstructs cloud-covered Sentinel-2 optical imagery by fusing:
- **Sentinel-2 Top-of-Atmosphere (TOA)** optical inputs (13 spectral bands: B1–B12, B8A)
- **Sentinel-1 Synthetic Aperture Radar (SAR)** inputs (VV + VH polarization)

It utilizes a **Modified DSen2-CR** deep neural network (18.95M parameters, 16 residual blocks with 128 feature channels, SAR feature fusion) trained on the **ALLClear** global benchmark dataset.

The system features an interactive, space-tech web dashboard with a real-time before/after draggable split slider, multi-modality viewers (Cloudy Optical, SAR Radar, Reconstructed Optical), dynamic cloud density filtering (0%–100%), SQLite persistent state management, and full geospatial GeoTIFF and PNG export.

---

## 2. System Architecture

```
                                  +---------------------------------------+
                                  |         HTML / CSS / JS Frontend      |
                                  |  - Before/After Draggable Slider      |
                                  |  - Sentinel-1 SAR & Optical Viewers   |
                                  |  - Cloud Density Threshold Filter     |
                                  |  - Metrics & Telemetry Inspector      |
                                  +-------------------|-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      FastAPI REST API Layer           |
                                  |  - /health  - /scenes  - /inference   |
                                  |  - /results - /metrics - /download    |
                                  +---------|-------------------|---------+
                                            |                   |
                     +----------------------+                   +---------------------+
                     v                                                                v
+------------------------------------------+                      +---------------------------------------+
|        SQLite & SQLAlchemy DB            |                      |   Phase 6 Geospatial Inference Engine |
|  - scenes         - inference_jobs       |                      |  - Tiled Smooth Cosine Blending       |
|  - results        - metrics              |                      |  - GeoTIFF Writer (13 Bands float32)  |
|  - model_versions - processing_history   |                      |  - PNG 4-Panel Comparison Renderer    |
+------------------------------------------+                      +-------------------|-------------------+
                                                                                      |
                                                                                      v
                                                                  +---------------------------------------+
                                                                  |  Modified DSen2-CR (best_model.pth)   |
                                                                  |  - 15 Input -> 13 Output Channels     |
                                                                  |  - NVIDIA GeForce RTX 4060 GPU        |
                                                                  +---------------------------------------+
```

---

## 3. Project Phase Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Machine & Dataset Audit | ✅ **COMPLETE** |
| **Phase 1** | Clean Foundation & Environment | ✅ **COMPLETE** |
| **Phase 2** | Preprocessing & Normalization Pipeline | ✅ **COMPLETE** |
| **Phase 3** | Modified DSen2-CR Model Architecture | ✅ **COMPLETE** |
| **Phase 4** | GPU Training Pipeline & Checkpointing | ✅ **COMPLETE** |
| **Phase 5** | Evaluation & Quantitative Benchmarks | ✅ **COMPLETE** |
| **Phase 6** | Production Geospatial Inference Pipeline | ✅ **COMPLETE** |
| **Phase 7** | Cloud-Density Layer & Scene Filtering | ✅ **COMPLETE** |
| **Phase 8** | FastAPI Production REST API | ✅ **COMPLETE** |
| **Phase 9** | SQLite Metadata Persistence & Audit Layer | ✅ **COMPLETE** |
| **Phase 10** | Professional Geospatial Web Frontend | ✅ **COMPLETE** |
| **Phase 11** | Full End-to-End System Integration | ✅ **COMPLETE** |

---

## 4. Quick Start & Execution

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA 12.1+ (e.g. RTX 4060)
- `uv` package manager

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/Kishore-mp45/CLOUD_RECONSTRUCTION_GDG.git
cd CLOUD_RECONSTRUCTION_GDG

# Synchronize dependencies with uv
uv sync
```

### 2. Launch the Application Server & Dashboard
```powershell
# Start the FastAPI server on port 8000
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### 3. Run Automated Integration Verification
```powershell
uv run python scripts/verify_system_integration.py
```

### 4. Run the Full Test Suite
```powershell
uv run pytest -v
```

---

## 5. Performance Benchmarks

- **Model Parameters**: 18,950,445 parameters (72.29 MB)
- **Model Checkpoint**: Best validation loss `0.1820` at Epoch 44
- **Test-Set Reconstruction Metrics**:
  - **Median PSNR**: `38.82 dB`
  - **Median SSIM**: `0.940`
  - **Median MAE**: `0.082`
  - **Median RMSE**: `0.115`
- **Inference Speed**: ~0.65s – 0.95s per standard scene tile on NVIDIA GeForce RTX 4060 GPU
- **Peak GPU VRAM**: ~0.893 GB VRAM

---

## 6. License
MIT License. Developed for GDG Geospatial Cloud Reconstruction Challenge.
