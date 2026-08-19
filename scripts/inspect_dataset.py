import os
import json
import subprocess
import sys
import glob
from pathlib import Path
from datetime import datetime

print("[PHASE 0] Environment audit")

audit = {
    "environment": {},
    "gpu": {},
    "cuda": {},
    "pytorch": {},
    "dependencies": {},
    "dataset": {},
    "metadata": {},
    "sentinel2": {},
    "sentinel1": {},
    "targets": {},
    "pairing": {},
    "temporal_matching": {},
    "geospatial": {},
    "geotiff": {},
    "data_quality": {},
    "missing_data": {},
    "earth_engine": {},
    "recommendations": {},
    "phase0_status": ""
}

# Environment
audit["environment"]["python_version"] = sys.version
audit["environment"]["os"] = os.name

print("[PHASE 0] NVIDIA/CUDA audit")
try:
    smi = subprocess.check_output(["nvidia-smi"], text=True)
    audit["gpu"]["nvidia-smi"] = "AVAILABLE"
except Exception as e:
    audit["gpu"]["nvidia-smi"] = f"FAILED: {e}"

try:
    nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
    audit["cuda"]["nvcc"] = "AVAILABLE"
except Exception as e:
    audit["cuda"]["nvcc"] = f"FAILED: {e}"

print("[PHASE 0] CUDA tensor test")
try:
    import torch
    audit["pytorch"]["version"] = torch.__version__
    audit["pytorch"]["cuda_version"] = torch.version.cuda
    audit["pytorch"]["cuda_is_available"] = torch.cuda.is_available()
    audit["pytorch"]["device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        audit["gpu"]["name"] = torch.cuda.get_device_name(0)
        t = torch.rand(3,3).cuda()
        audit["pytorch"]["tensor_test"] = "SUCCESS"
        audit["cuda"]["status"] = "READY"
    else:
        audit["pytorch"]["tensor_test"] = "FAILED"
        audit["cuda"]["status"] = "BLOCKED"
except Exception as e:
    audit["pytorch"]["error"] = str(e)
    audit["cuda"]["status"] = "BLOCKED"

print("[PHASE 0] Dependency audit")
deps = ["rasterio", "osgeo.gdal", "numpy", "scipy", "tifffile", "pyproj", "shapely", "geopandas", "xarray", "rioxarray"]
for dep in deps:
    try:
        if dep == "osgeo.gdal":
            import osgeo.gdal
            audit["dependencies"]["GDAL"] = "INSTALLED"
        else:
            __import__(dep)
            audit["dependencies"][dep] = "INSTALLED"
    except ImportError:
        audit["dependencies"][dep] = "MISSING"

print("[PHASE 0] Dataset inspection")
dataset_root = Path("allclear_dataset")
if dataset_root.exists():
    s2_files = list(dataset_root.glob("roi*/**/s2_toa/*.tif"))
    s1_files = list(dataset_root.glob("roi*/**/s1/*.tif"))
    rois = set(p.parent.parent.parent.name for p in s2_files)
    
    audit["dataset"]["rois"] = len(rois)
    audit["dataset"]["s2_files"] = len(s2_files)
    audit["dataset"]["s1_files"] = len(s1_files)
else:
    audit["dataset"]["status"] = "NOT_FOUND"

print("[PHASE 0] Metadata parsing")
meta_path = Path("allclear_test_metadata.json")
if meta_path.exists():
    with open(meta_path, "r") as f:
        meta = json.load(f)
    audit["metadata"]["record_count"] = len(meta)
    
    target_count = 0
    missing_s1 = 0
    missing_s2 = 0
    s2_dates = []
    s1_dates = []
    target_dates = []
    temporal_diffs_s1 = []
    temporal_diffs_s2 = []
    
    for key, val in meta.items():
        if "target" in val and len(val["target"]) > 0:
            target_count += 1
            t_path = val["target"][0]
            try:
                t_date = datetime.strptime(t_path.split("_")[-2], "%Y%m%d") # Need exact logic based on filename
            except:
                pass
        
        if "s2_toa" not in val or len(val["s2_toa"]) == 0:
            missing_s2 += 1
        
        if "s1" not in val or len(val["s1"]) == 0:
            missing_s1 += 1
            
    audit["targets"]["target_count"] = target_count
    audit["missing_data"]["missing_s1"] = missing_s1
    audit["missing_data"]["missing_s2"] = missing_s2

print("[PHASE 0] GeoTIFF inspection")
try:
    import rasterio
    import numpy as np
    
    if dataset_root.exists() and len(s2_files) > 0:
        sample_s2 = s2_files[0]
        with rasterio.open(sample_s2) as src:
            audit["geotiff"]["s2_sample_path"] = str(sample_s2)
            audit["geotiff"]["s2_crs"] = str(src.crs)
            audit["geotiff"]["s2_width"] = src.width
            audit["geotiff"]["s2_height"] = src.height
            audit["geotiff"]["s2_bands"] = src.count
            audit["geotiff"]["s2_dtypes"] = [str(d) for d in src.dtypes]
            audit["geotiff"]["s2_descriptions"] = src.descriptions
            audit["geotiff"]["s2_transform"] = str(src.transform)
            
    if dataset_root.exists() and len(s1_files) > 0:
        sample_s1 = s1_files[0]
        with rasterio.open(sample_s1) as src:
            audit["geotiff"]["s1_sample_path"] = str(sample_s1)
            audit["geotiff"]["s1_crs"] = str(src.crs)
            audit["geotiff"]["s1_width"] = src.width
            audit["geotiff"]["s1_height"] = src.height
            audit["geotiff"]["s1_bands"] = src.count
            audit["geotiff"]["s1_dtypes"] = [str(d) for d in src.dtypes]
            audit["geotiff"]["s1_descriptions"] = src.descriptions
            audit["geotiff"]["s1_transform"] = str(src.transform)
            
    audit["data_quality"]["status"] = "INSPECTED"
except ImportError:
    audit["geotiff"]["status"] = "RASTERIO_MISSING"
    audit["data_quality"]["status"] = "SKIPPED_DUE_TO_MISSING_DEPS"


print("[PHASE 0] S1/S2 temporal matching")
# Fallback since full parsing requires exact filename formats
audit["temporal_matching"]["status"] = "AMBIGUOUS - Dates must be parsed from manifest keys and paths"
audit["pairing"]["status"] = "TRAINING RELATIONSHIP AMBIGUOUS - target and input dates differ"

audit["earth_engine"]["considerations"] = "Earth Engine not implemented in Phase 0. Future integration will require downloading S1/S2 collections, matching CRS/resolution to this dataset, and applying DSen2-CR on the fly."
audit["recommendations"]["phase1"] = "1. Establish exact dataset loaders reading from allclear_test_metadata.json. 2. Handle missing S1 inputs. 3. Normalize S1 and S2 inputs."

print("[PHASE 0] Writing audit")
with open("docs/dataset_audit.json", "w") as f:
    json.dump(audit, f, indent=4)

print("[PHASE 0] COMPLETE")
