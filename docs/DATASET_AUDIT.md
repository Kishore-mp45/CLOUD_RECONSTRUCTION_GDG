# ALLCLEAR DATASET AUDIT (PHASE 0)

## Executive Summary
This document provides a read-only audit of the AllClear test dataset, machine environment, and PyTorch/CUDA configuration.

## 1. Machine/Python/PyTorch/CUDA/GPU
- **OS**: Windows (nt)
- **Python Version**: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **NVIDIA-SMI**: AVAILABLE
- **CUDA Toolkit (nvcc)**: FAILED: [WinError 2] The system cannot find the file specified (Not strictly required as PyTorch brings its own binaries)
- **PyTorch Version**: 2.5.1+cu121
- **CUDA Version in PyTorch**: 12.1
- **CUDA Status**: READY (Tensor test succeeded on device 0)

## 2. Dependency Status
- **rasterio**: INSTALLED
- **numpy**: INSTALLED
- **osgeo.gdal**: MISSING
- **scipy**: MISSING
- **tifffile**: MISSING
- **pyproj**: MISSING
- **shapely**: MISSING
- **geopandas**: MISSING
- **xarray**: MISSING
- **rioxarray**: MISSING

## 3. Dataset Structure/Statistics
- **ROIs**: 3698
- **Total S2 Files**: 14792
- **Total S1 Files**: 4608
- **Missing S1 files in metadata**: 1203 (out of 3698)
- **Missing S2 files in metadata**: 0

## 4. Metadata
- **Total Records**: 3698
- **Target Count**: 3698

## 5. GeoTIFF Properties

### Sentinel-2 (S2) Example: `allclear_dataset\roi100011\2022_3\s2_toa\roi100011_s2_toa_2022_3_28_median.tif`
- **CRS**: EPSG:32722
- **Dimensions**: 308 x 308
- **Bands**: 13
- **Data Types**: float64
- **Band Names**: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12

### Sentinel-1 (S1) Example: `allclear_dataset\roi100663\2022_9\s1\roi100663_s1_2022_9_13_median.tif`
- **CRS**: EPSG:32755
- **Dimensions**: 313 x 313
- **Bands**: 2
- **Data Types**: float64
- **Band Names**: VV, VH

## 6. S2/S1/target pairing & Temporal Matching
- **Target Pairing**: TRAINING RELATIONSHIP AMBIGUOUS - target and input dates differ
- **Temporal Matching**: AMBIGUOUS - Dates must be parsed from manifest keys and paths

## 7. Missing Data
- **Missing S1**: 1203 records do not have an S1 input

## 8. Earth Engine Considerations
Earth Engine not implemented in Phase 0. Future integration will require downloading S1/S2 collections, matching CRS/resolution to this dataset, and applying DSen2-CR on the fly.

## 9. Recommended Phase 1 Actions
1. Establish exact dataset loaders reading from allclear_test_metadata.json.
2. Handle missing S1 inputs.
3. Normalize S1 and S2 inputs.
