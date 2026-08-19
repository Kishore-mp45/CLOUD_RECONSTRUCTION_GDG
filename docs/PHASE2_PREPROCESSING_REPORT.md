# PHASE 2 PREPROCESSING REPORT

## Summary

The Phase 2 preprocessing pipeline has successfully completed. This phase converted the raw ALLClear dataset into a clean, reproducible, SAR-supervised training dataset suitable for the DSen2-CR architecture.

## Execution Details
- **Dataset Root:** `D:\allclear_test_proi1_v1`
- **Elapsed Time:** 1749.2 seconds (~29 minutes)

## Validation and Filtering
The pipeline evaluated the raw metadata against strict rules (existence of S2 and S1 data).
- **Total Metadata Records:** 3698
- **Valid SAR-supervised Pairs:** 2495
- **Rejected Pairs:** 1203
- **Primary Rejection Reason:** No S1 observation in metadata.

## Triplet Expansion
The valid pairs were expanded to include multiple S2 input dates per target date (if available).
- **Total Valid Triplets:** 7485

## Data Splits
Triplets were split based on Region of Interest (ROI) to prevent data leakage.
- **Leakage Check:** PASS (Zero ROI overlap across all splits)
- **Train ROIs / Triplets:** 1996 / 5988
- **Val ROIs / Triplets:** 250 / 750
- **Test ROIs / Triplets:** 249 / 747

## Normalization
Normalization statistics were computed using a subset of the training set (500 triplets) and saved to `data/normalization/normalization.json`.

## Patch Generation
256x256 image patches were extracted from the valid triplets, keeping those with ≤5% nodata fraction.
- **Patch Size:** 256
- **Patch Stride:** 128
- **Total Valid Patches Generated:** 7400
- **Total Rejected Patches (nodata > 5%):** 85

### Final Patch Counts
- **Train Patches:** 5916
- **Val Patches:** 741
- **Test Patches:** 743

## Outputs
- **Manifests Directory:** `data/manifests/`
- **Train Manifest:** `data/manifests/train.json`
- **Val Manifest:** `data/manifests/val.json`
- **Test Manifest:** `data/manifests/test.json`
- **Normalization Stats:** `data/normalization/normalization.json`
- **Rejection Log:** `data/manifests/rejected_records.json`

## Next Steps
With clean training patches and normalization statistics generated, the dataset is fully prepared for Phase 3 (Training Loop + Metrics).
