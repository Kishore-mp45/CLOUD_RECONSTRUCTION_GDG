# frontend/

HTML + CSS + JavaScript frontend — Phase 3+

This directory will contain the static frontend files:
- index.html
- css/
- js/

The frontend will display:
- Original cloudy Sentinel-2 image
- Sentinel-1 SAR image
- Reconstructed (cloud-removed) Sentinel-2 image
- Before/after slider
- Metrics: PSNR, SSIM, MAE, RMSE, inference time
- Image metadata
- Model information
- Download: reconstructed GeoTIFF
- Download: PNG preview

The frontend does NOT include training/evaluation dashboards.
The frontend NEVER triggers model retraining.
