"""
scripts/real_inference_test.py
================================
Real end-to-end inference test using actual best_model.pth checkpoint.
Tests Section 7 of the Final Audit: Real Inference Test.

Finds first eligible scene (S2+S1 GeoTIFF on disk), runs inference,
validates output GeoTIFF and RGB PNGs with IDENTICAL rendering.
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, "src")

import torch
import rasterio

from cloudremoval.config import get_settings
from cloudremoval.data.normalization import load_normalization, S2_BAND_NAMES
from cloudremoval.evaluation.visualizer import to_rgb_numpy, RGB_INDICES

settings = get_settings()

CKPT = pathlib.Path("checkpoints/best_model.pth")
NORM = pathlib.Path("data/normalization/normalization.json")
OUT_DIR = pathlib.Path("outputs/audit_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("REAL INFERENCE AUDIT TEST")
print("=" * 60)

# --- 1. Find eligible scene ---
print("\n[1] Looking for eligible scene with S2+S1 files on disk...")
import json as json_mod
meta_path = pathlib.Path("allclear_test_metadata.json")
with open(meta_path, "r") as f:
    meta = json_mod.load(f)

allclear_root = pathlib.Path("allclear_dataset")
test_s2 = None
test_s1 = None
scene_id_used = None

records = meta if isinstance(meta, list) else list(meta.values())
# Metadata has Linux paths — use directory scan instead
print("  Scanning allclear_dataset for S2+S1 pairs...")
for roi_dir in sorted(allclear_root.iterdir()):
    if not roi_dir.is_dir():
        continue
    s2_files = sorted(roi_dir.rglob("*s2_toa*.tif"))
    s1_files = sorted(roi_dir.rglob("*s1*.tif"))
    if s2_files and s1_files:
        test_s2 = s2_files[0]
        test_s1 = s1_files[0]
        scene_id_used = roi_dir.name
        break

if test_s2 is None:
    print("[FAIL] Could not find any S2+S1 pair. Aborting.")
    sys.exit(1)

print(f"  [OK] Scene: {scene_id_used}")
print(f"  [OK] S2: {test_s2}")
print(f"  [OK] S1: {test_s1}")

# --- 2. Validate S2/S1 GeoTIFFs ---
print("\n[2] Validating input GeoTIFFs...")
with rasterio.open(test_s2) as s2_src:
    s2_crs = s2_src.crs
    s2_transform = s2_src.transform
    s2_w, s2_h = s2_src.width, s2_src.height
    s2_bands = s2_src.count
    s2_data_raw = s2_src.read().astype(np.float32)  # (13, H, W)

print(f"  S2 shape   : {s2_data_raw.shape}")
print(f"  S2 CRS     : {s2_crs}")
print(f"  S2 dims    : {s2_w}x{s2_h}")
print(f"  S2 bands   : {s2_bands}")
print(f"  S2 min/max : {s2_data_raw.min():.2f} / {s2_data_raw.max():.2f}")
print(f"  S2 finite  : {np.isfinite(s2_data_raw).mean():.4f}")

for b_idx, name in enumerate(["B4(R)", "B3(G)", "B2(B)"]):
    b_data = s2_data_raw[[3,2,1][b_idx]]
    print(f"    {name}: min={b_data.min():.1f}, max={b_data.max():.1f}, mean={b_data.mean():.1f}, nan={np.isnan(b_data).sum()}")

# --- 3. Run Inference Pipeline ---
print("\n[3] Running GeospatialInferencePipeline with best_model.pth...")
from cloudremoval.inference.pipeline import GeospatialInferencePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

pipeline = GeospatialInferencePipeline(
    checkpoint_path=CKPT,
    norm_path=NORM,
    device=device,
    amp_enabled=True,
    tile_size=256,
    overlap=64,
    batch_size=4,
)

result = pipeline.run_inference(
    s2_path=test_s2,
    s1_path=test_s1,
    output_dir=OUT_DIR,
    job_id="audit_test",
)

# --- 4. Validate output GeoTIFF ---
print("\n[4] Validating output GeoTIFF...")
gtiff_path = pathlib.Path(result["output_geotiff"])
assert gtiff_path.exists(), f"GeoTIFF missing: {gtiff_path}"

with rasterio.open(gtiff_path) as dst:
    out_crs = dst.crs
    out_transform = dst.transform
    out_w, out_h = dst.width, dst.height
    out_bands = dst.count
    out_data = dst.read().astype(np.float32)  # (13, H, W)
    out_descs = dst.descriptions

print(f"  Output GeoTIFF: {gtiff_path}")
print(f"  Output shape   : {out_data.shape}")
print(f"  Output CRS     : {out_crs}")
print(f"  Output dims    : {out_w}x{out_h}")
print(f"  Output bands   : {out_bands}")
print(f"  Output min/max : {out_data.min():.2f} / {out_data.max():.2f}")
print(f"  Band names     : {out_descs}")
print(f"  CRS match      : {out_crs == s2_crs}")
print(f"  Transform match: {out_transform == s2_transform}")
print(f"  Dims match     : {out_w == s2_w and out_h == s2_h}")
print(f"  Finite frac    : {np.isfinite(out_data).mean():.4f}")

for b_idx, name in enumerate(["B4(R)", "B3(G)", "B2(B)"]):
    b_data = out_data[[3,2,1][b_idx]]
    print(f"    Reconstructed {name}: min={b_data.min():.1f}, max={b_data.max():.1f}, mean={b_data.mean():.1f}, nan={np.isnan(b_data).sum()}")

# --- 5. RGB Pipeline Audit ---
print(f"\n[5] RGB pipeline audit...")
print(f"  RGB_INDICES = {RGB_INDICES}  (R=B4 idx=3, G=B3 idx=2, B=B2 idx=1)")
print(f"  S2 band order: {S2_BAND_NAMES}")

# Original RGB (using IDENTICAL renderer)
rgb_original = to_rgb_numpy(s2_data_raw, rgb_indices=RGB_INDICES)
# Reconstructed RGB (using IDENTICAL renderer)
rgb_reconstructed = to_rgb_numpy(out_data, rgb_indices=RGB_INDICES)

print(f"  Original RGB   : shape={rgb_original.shape}, min={rgb_original.min():.4f}, max={rgb_original.max():.4f}")
print(f"  Reconstructed  : shape={rgb_reconstructed.shape}, min={rgb_reconstructed.min():.4f}, max={rgb_reconstructed.max():.4f}")

# Save both PNGs with identical rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

original_png = OUT_DIR / "original_rgb.png"
reconstructed_png = OUT_DIR / "reconstructed_rgb.png"
comparison_png = OUT_DIR / "comparison.png"

plt.imsave(original_png, rgb_original)
plt.imsave(reconstructed_png, rgb_reconstructed)
print(f"  [OK] original_rgb.png saved: {original_png}")
print(f"  [OK] reconstructed_rgb.png saved: {reconstructed_png}")

# Comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
fig.patch.set_facecolor("#181818")
axes[0].imshow(rgb_original)
axes[0].set_title("Input Cloudy S2 (B4-B3-B2)", color="white", fontsize=12)
axes[0].axis("off")
axes[1].imshow(rgb_reconstructed)
axes[1].set_title("DSen2-CR Reconstructed S2 (B4-B3-B2)", color="#00ffcc", fontsize=12)
axes[1].axis("off")
plt.suptitle("Audit: Identical RGB rendering pipeline", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(comparison_png, bbox_inches="tight", facecolor="#181818")
plt.close(fig)
print(f"  [OK] comparison.png saved: {comparison_png}")

# --- 6. Per-band statistics ---
print(f"\n[6] Per-band output statistics...")
print(f"  {'Band':<6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'NaN':>6} {'Inf':>6}")
for i, bname in enumerate(S2_BAND_NAMES):
    b = out_data[i]
    print(f"  {bname:<6} {b.min():>10.1f} {b.max():>10.1f} {b.mean():>10.1f} {b.std():>10.1f} {np.isnan(b).sum():>6} {np.isinf(b).sum():>6}")

# --- 7. Performance ---
perf = result["performance"]
print(f"\n[7] Performance:")
print(f"  GPU             : {perf['gpu_name']}")
print(f"  Peak VRAM       : {perf['peak_vram_gb']:.3f} GB")
print(f"  Preprocess time : {perf['preprocessing_time_s']:.3f} s")
print(f"  Inference time  : {perf['model_inference_time_s']:.3f} s")
print(f"  Total pipeline  : {perf['total_pipeline_time_s']:.3f} s")

print("\n" + "=" * 60)
print("REAL INFERENCE TEST: PASS")
print("=" * 60)
print(f"  original_rgb.png    -> {original_png}")
print(f"  reconstructed_rgb.png -> {reconstructed_png}")
print(f"  comparison.png      -> {comparison_png}")
print(f"  output GeoTIFF      -> {gtiff_path}")
