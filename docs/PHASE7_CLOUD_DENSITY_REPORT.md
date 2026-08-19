# PHASE 7 — CLOUD-DENSITY LAYER REPORT

**Generated:** 2026-08-20
**Phase:** Phase 7 (Cloud-Density Layer)
**Status:** COMPLETE & VERIFIED

---

## 1. Executive Summary

Phase 7 implements a modular, reusable **Cloud-Density Analysis & Filtering Layer**. This layer sits between raw satellite scene ingestion and cloud removal inference. It validates per-pixel cloud probabilities, calculates exact Region of Interest (ROI) cloud-density percentages, and applies dual-stage configurable filtering (pixel threshold and scene threshold).

---

## 2. Cloud Density Mathematical Definition

### A. Pixel-Level Cloud Probability
Satellite optical scenes provide per-pixel cloud confidence scores ($P(x, y) \in [0.0, 100.0]\%$).
- Default Pixel Cloud Threshold: **$\theta_{\text{pixel}} = 60.0\%$**
- A valid pixel at $(x, y)$ is classified as **cloudy** if:
  $$\text{IsCloudy}(x, y) = \begin{cases} 1 & \text{if } P(x, y) \ge \theta_{\text{pixel}} \\ 0 & \text{otherwise} \end{cases}$$

### B. ROI-Level Scene Cloud Density Percentage
Given an arbitrary spatial ROI and a valid/non-nodata pixel mask:
- Active Valid Pixels in ROI: $N_{\text{valid}} = \sum_{(x, y) \in \text{ROI}} \text{IsValid}(x, y)$
- Cloudy Valid Pixels in ROI: $N_{\text{cloudy}} = \sum_{(x, y) \in \text{ROI}} \left( \text{IsValid}(x, y) \land \text{IsCloudy}(x, y) \right)$
- **Scene Cloud Density:**
  $$\text{Cloud Density (\%)} = \left( \frac{N_{\text{cloudy}}}{N_{\text{valid}}} \right) \times 100$$

### C. Scene-Level Eligibility Filter
- Default Scene Cloud-Density Threshold: **$\Theta_{\text{scene}} = 60.0\%$**
  - If $\text{Cloud Density (\%)} \ge \Theta_{\text{scene}} \implies$ **ELIGIBLE** (Requires cloud removal)
  - If $\text{Cloud Density (\%)} < \Theta_{\text{scene}} \implies$ **FILTERED OUT** (Already sufficiently clear)

---

## 3. Architecture & Modularity

```
           Sentinel-2 Scene / Earth Engine Source
                            ↓
               validate_cloud_probability()
             (Shape, Finite, Range [0, 100]%)
                            ↓
                calculate_cloud_density()
          (ROI Masking + Valid-Pixel Accounting)
                            ↓
                      analyze_scene()
            (Compare against Scene Threshold)
             ↙                             ↘
       [ELIGIBLE]                     [FILTERED]
(Added to processing queue)     (Excluded from inference)
```

---

## 4. Multi-Condition Demonstration Results

Using `scripts/analyze_cloud_density.py --demo`, representative scenes were analyzed:

| Scene ID | Condition | Mean Cloud Prob | Cloud Density (%) | Status | Result |
|---|---|---|---|---|---|
| `DEMO_SCENE_01` | Clear Sky | 12.49% | **0.0%** | **FILTERED** | Correctly rejected |
| `DEMO_SCENE_02` | Moderate Cumulus | 37.12% | **30.6%** | **FILTERED** | Correctly below 60% threshold |
| `DEMO_SCENE_03` | Dense Monsoon | 85.00% | **100.0%** | **ELIGIBLE** | Correctly selected for cloud removal |
| `DEMO_SCENE_04` | Partial Nodata ROI | 80.01% | **100.0%** | **ELIGIBLE** | Valid pixels correctly accounted |
| `DEMO_SCENE_05` | Corrupt / All NaNs | N/A | **0.0%** | **ERROR** | Non-finite data trapped cleanly |

Output JSON artifact: [`outputs/cloud/cloud_density_results.json`](file:///d:/allclear_test_proi1_v1/outputs/cloud/cloud_density_results.json)

---

## 5. Edge Cases & Robustness

- **Missing / None Arrays:** Raises `CloudDataValidationError`.
- **All NaNs / Infs:** Trapped and recorded with `status: error` without crashing the batch pipeline.
- **Fractional Scale ([0.0, 1.0]):** Automatically rescaled to percentage ([0.0, 100.0]%).
- **Empty / Zero-pixel ROI:** Trapped with `InvalidROIDataError`.
- **Nodata Pixel Isolation:** Nodata pixels are masked out so they never distort the cloud density calculation.

---

## 6. Google Earth Engine (Phase 8 Integration Contract)

In Phase 8 (Earth Engine live ingestion), the cloud-density module integrates seamlessly with the following adapter design:

```python
# Earth Engine Adapter Contract
ee_cloud_prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
# Download ROI array -> calculate_cloud_density(cloud_prob, roi_mask) -> filter_scene()
```

---

## 7. Test Suite Verification

- **15 / 15 tests passed** in `tests/cloud/test_cloud_density.py`
- **189 / 189 total tests passed** across the entire repository (100% pass rate).
