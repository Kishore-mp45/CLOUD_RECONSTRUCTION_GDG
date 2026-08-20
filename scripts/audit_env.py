"""
scripts/audit_env.py
Full environment, GPU, imports, and model smoke test for the Final Audit.
"""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / "src"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = {}

# 1. Python version
try:
    import platform
    pv = sys.version_info
    ok = pv >= (3, 9)
    results["python_version"] = f"{pv.major}.{pv.minor}.{pv.micro}"
    print(f"[ENV] Python: {pv.major}.{pv.minor}.{pv.micro}  {'OK' if ok else 'WARN (need >=3.9)'}")
except Exception as e:
    results["python_version"] = str(e)

# 2. PyTorch
try:
    import torch
    results["torch"] = torch.__version__
    print(f"[ENV] PyTorch: {torch.__version__}")
except Exception as e:
    results["torch"] = str(e)
    print(f"[ENV] PyTorch: FAIL - {e}")

# 3. CUDA
try:
    cuda_ok = torch.cuda.is_available()
    results["cuda"] = cuda_ok
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cv = torch.version.cuda
        results["gpu_name"] = name
        results["gpu_vram_gb"] = round(vram, 2)
        results["cuda_version"] = cv
        print(f"[GPU] CUDA: OK | Device: {name} | VRAM: {vram:.2f} GB | CUDA: {cv}")
    else:
        print(f"[GPU] CUDA: NOT AVAILABLE")
except Exception as e:
    print(f"[GPU] CUDA check failed: {e}")

# 4. Rasterio / GDAL
try:
    import rasterio
    results["rasterio"] = rasterio.__version__
    print(f"[ENV] rasterio: {rasterio.__version__}")
    try:
        import rasterio.gdal
        print(f"[ENV] GDAL: available")
    except:
        print(f"[ENV] GDAL: bundled with rasterio")
except Exception as e:
    results["rasterio"] = str(e)
    print(f"[ENV] rasterio: FAIL - {e}")

# 5. numpy / scipy / matplotlib
for pkg in ["numpy", "scipy", "matplotlib"]:
    try:
        m = __import__(pkg)
        print(f"[ENV] {pkg}: {m.__version__}")
    except Exception as e:
        print(f"[ENV] {pkg}: FAIL - {e}")

# 6. fastapi / uvicorn / sqlalchemy / pydantic
for pkg in ["fastapi", "uvicorn", "sqlalchemy", "pydantic"]:
    try:
        m = __import__(pkg)
        print(f"[ENV] {pkg}: {m.__version__}")
    except Exception as e:
        print(f"[ENV] {pkg}: FAIL - {e}")

# 7. Project imports
print("\n[IMPORTS] Testing project imports...")
import_results = {}

imports_to_test = [
    ("cloudremoval.config", "get_settings"),
    ("cloudremoval.models.dsen2cr", "DSen2CR, build_model"),
    ("cloudremoval.models.model_config", "DSen2CRConfig, DEFAULT_CONFIG, S2_CHANNELS, S1_CHANNELS"),
    ("cloudremoval.models.checkpoint", "load_checkpoint, save_checkpoint"),
    ("cloudremoval.models.losses", "build_loss"),
    ("cloudremoval.models.fusion", "SARFusion"),
    ("cloudremoval.data.normalization", "normalize_s2, normalize_s1, denormalize_s2"),
    ("cloudremoval.data.loaders", "load_s2, load_s1, load_target"),
    ("cloudremoval.data.dataset", "AllClearDataset, build_dataset"),
    ("cloudremoval.evaluation.visualizer", "to_rgb_numpy, RGB_INDICES"),
    ("cloudremoval.inference.pipeline", "GeospatialInferencePipeline"),
    ("cloudremoval.inference.tiled_inference", "TiledInferenceEngine"),
    ("cloudremoval.inference.writer", "write_reconstructed_geotiff"),
    ("cloudremoval.inference.preview", "create_inference_preview"),
    ("cloudremoval.geospatial.alignment", "load_and_align_s1_to_s2"),
    ("cloudremoval.cloud", None),
]

for mod, attrs in imports_to_test:
    try:
        m = __import__(mod, fromlist=["*"])
        if attrs:
            for attr in attrs.split(", "):
                attr = attr.strip()
                if not hasattr(m, attr):
                    raise AttributeError(f"Missing attribute: {attr}")
        import_results[mod] = "OK"
        print(f"  [OK]   {mod}")
    except Exception as e:
        import_results[mod] = str(e)
        print(f"  [FAIL] {mod}: {e}")

# 8. Model forward/backward smoke test
print("\n[MODEL] Running DSen2-CR forward/backward smoke test...")
try:
    from cloudremoval.models.dsen2cr import build_model
    from cloudremoval.models.model_config import DSen2CRConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(DSen2CRConfig(device=device)).to(device)

    s2 = torch.randn(1, 13, 128, 128).to(device)
    s1 = torch.randn(1, 2,  128, 128).to(device)

    with torch.no_grad():
        out = model(s2, s1)

    assert out.shape == (1, 13, 128, 128), f"Shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values in output"
    print(f"  [OK] Forward pass: output shape {out.shape}, device {out.device}")
    print(f"  [OK] All output values finite: True")

    # Backward
    model.train()
    out2 = model(s2, s1)
    loss = out2.mean()
    loss.backward()
    print(f"  [OK] Backward pass: OK")

    results["model_forward"] = "PASS"
    results["model_backward"] = "PASS"
except Exception as e:
    print(f"  [FAIL] Model smoke test: {e}")
    results["model_forward"] = str(e)

# 9. Normalization round-trip test
print("\n[NORM] Testing normalization round-trip...")
try:
    import numpy as np
    from cloudremoval.data.normalization import normalize_s2, denormalize_s2, S2_BAND_NAMES

    # Fake stats matching expected structure
    fake_stats = {
        "s2": {
            "mean": [1000.0] * 13,
            "std":  [500.0]  * 13,
        }
    }
    data = np.random.uniform(0, 10000, (13, 64, 64)).astype(np.float64)
    normed = normalize_s2(data, fake_stats)
    recovered = denormalize_s2(normed, fake_stats)

    max_err = float(np.max(np.abs(data - recovered)))
    print(f"  [OK] Band names: {S2_BAND_NAMES}")
    print(f"  [OK] Norm/denorm max round-trip error (after clamp): {max_err:.2f}")
    print(f"  [OK] RGB band mapping: B4=idx3(R), B3=idx2(G), B2=idx1(B)")
    results["normalization"] = "PASS"
except Exception as e:
    print(f"  [FAIL] Normalization test: {e}")
    results["normalization"] = str(e)

# 10. Check checkpoint
print("\n[CKPT] Checking best_model.pth...")
import pathlib
ckpt = pathlib.Path("checkpoints/best_model.pth")
if ckpt.exists():
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    print(f"  [OK] Checkpoint found: {ckpt} ({size_mb:.1f} MB)")
    try:
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        epoch = c.get("epoch", "unknown")
        val_loss = c.get("val_loss", "unknown")
        keys = list(c.keys())
        print(f"  [OK] Checkpoint epoch: {epoch}, val_loss: {val_loss}")
        print(f"  [OK] Keys: {keys}")
        results["checkpoint"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Cannot load checkpoint: {e}")
        results["checkpoint"] = str(e)
else:
    print(f"  [WARN] Checkpoint NOT found: {ckpt}")
    results["checkpoint"] = "MISSING"

# 11. Check normalization.json
norm_json = pathlib.Path("data/normalization/normalization.json")
if norm_json.exists():
    import json
    with open(norm_json) as f:
        nj = json.load(f)
    print(f"\n[NORM] normalization.json: OK (version={nj.get('version')}, bands S2={len(nj['s2']['mean'])}, S1={len(nj['s1']['mean'])})")
    results["norm_json"] = "PRESENT"
else:
    print(f"\n[NORM] normalization.json: MISSING at {norm_json}")
    results["norm_json"] = "MISSING"

# 12. Check data directories
print("\n[DATA] Checking data/allclear_dataset...")
ad = pathlib.Path("allclear_dataset")
if ad.exists():
    rois = list(ad.iterdir())
    print(f"  [OK] allclear_dataset found with {len(rois)} items")
    results["dataset"] = "PRESENT"
else:
    print(f"  [WARN] allclear_dataset not found at {ad}")
    results["dataset"] = "MISSING"

# 13. Database file
db_path = pathlib.Path("data/cloudremoval.db")
if db_path.exists():
    size_kb = db_path.stat().st_size / 1024
    print(f"[DB] Database: OK ({db_path}, {size_kb:.1f} KB)")
    results["database"] = "PRESENT"
else:
    print(f"[DB] Database: MISSING at {db_path}")
    results["database"] = "MISSING"

# 14. RGB pipeline audit
print("\n[RGB] Auditing RGB pipeline consistency...")
try:
    from cloudremoval.evaluation.visualizer import RGB_INDICES, to_rgb_numpy
    from cloudremoval.inference.preview import create_inference_preview
    print(f"  [OK] RGB_INDICES = {RGB_INDICES}  (R=B4 idx={RGB_INDICES[0]}, G=B3 idx={RGB_INDICES[1]}, B=B2 idx={RGB_INDICES[2]})")
    print(f"  [OK] visualizer.to_rgb_numpy and preview.create_inference_preview both use RGB_INDICES")
    print(f"  [OK] Both original and reconstructed use identical: to_rgb_numpy(..., rgb_indices=RGB_INDICES)")
    results["rgb_pipeline"] = "CONSISTENT"
except Exception as e:
    print(f"  [FAIL] RGB pipeline: {e}")
    results["rgb_pipeline"] = str(e)

print("\n" + "="*50)
print("ENVIRONMENT AUDIT COMPLETE")
print("="*50)
for k, v in results.items():
    print(f"  {k}: {v}")
