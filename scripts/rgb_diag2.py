"""
scripts/rgb_diag2.py
Full pipeline RGB diagnostic (ASCII-safe for Windows cp1252).
"""
import sys, json, pathlib, numpy as np, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "src")

import torch, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEP = "=" * 60

# Find scene
root = pathlib.Path("allclear_dataset")
test_s2, test_s1 = None, None
for roi_dir in sorted(root.iterdir()):
    if not roi_dir.is_dir(): continue
    s2s = sorted(roi_dir.rglob("*s2_toa*.tif"))
    s1s = sorted(roi_dir.rglob("*s1*.tif"))
    if s2s and s1s:
        test_s2, test_s1 = s2s[0], s1s[0]
        break

print(f"[SCENE] S2: {test_s2}")
print(f"[SCENE] S1: {test_s1}")

# ---- STEP 1: Raw S2 ----
print(f"\n{SEP}\nSTEP 1: RAW S2 INPUT\n{SEP}")
with rasterio.open(test_s2) as src:
    s2_raw = src.read().astype(np.float32)
    s2_descs = src.descriptions

s2_band_names = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
print(f"  Shape: {s2_raw.shape}  NaN: {np.isnan(s2_raw).sum()}")
print(f"  Band descriptions from file: {s2_descs}")
print(f"  Global min/max (finite): {np.nanmin(s2_raw):.1f} / {np.nanmax(s2_raw):.1f}")
print()
print(f"  {'Band':<6} {'Mean':>10} {'Max':>10}")
for i,bn in enumerate(s2_band_names):
    b = s2_raw[i]
    print(f"  {bn:<6} {np.nanmean(b):>10.1f} {np.nanmax(b):>10.1f}")

print(f"\n  RGB BANDS:")
print(f"  R=B4 idx=3: mean={np.nanmean(s2_raw[3]):.1f}  max={np.nanmax(s2_raw[3]):.1f}")
print(f"  G=B3 idx=2: mean={np.nanmean(s2_raw[2]):.1f}  max={np.nanmax(s2_raw[2]):.1f}")
print(f"  B=B2 idx=1: mean={np.nanmean(s2_raw[1]):.1f}  max={np.nanmax(s2_raw[1]):.1f}")

# ---- STEP 2: Normalization stats ----
print(f"\n{SEP}\nSTEP 2: NORMALIZATION STATS\n{SEP}")
norm_path = pathlib.Path("data/normalization/normalization.json")
with open(norm_path) as f:
    norm_stats = json.load(f)
s2_means = norm_stats["s2"]["mean"]
s2_stds  = norm_stats["s2"]["std"]
print(f"  Z_clamp: {norm_stats.get('z_clamp', 5.0)}")
print(f"  {'Band':<6} {'Mean':>10} {'Std':>10}")
for i,bn in enumerate(s2_band_names):
    print(f"  {bn:<6} {s2_means[i]:>10.2f} {s2_stds[i]:>10.2f}")

# ---- STEP 3: Load, normalize, run model ----
print(f"\n{SEP}\nSTEP 3: PIPELINE EXECUTION\n{SEP}")
from cloudremoval.geospatial.alignment import load_and_align_s1_to_s2
from cloudremoval.data.normalization import normalize_s2, normalize_s1, denormalize_s2
from cloudremoval.models.dsen2cr import build_model
from cloudremoval.models.model_config import DSen2CRConfig
from cloudremoval.models import load_checkpoint

s2_aligned, s1_aligned, s2_meta = load_and_align_s1_to_s2(test_s2, test_s1)
print(f"  S2 aligned: {s2_aligned.shape}, NaN: {np.isnan(s2_aligned).sum()}")

s2_norm = normalize_s2(s2_aligned, norm_stats)
s1_norm = normalize_s1(s1_aligned, norm_stats)
print(f"  S2 norm range: [{s2_norm.min():.3f}, {s2_norm.max():.3f}]")
print(f"  S2 norm RGB: R={s2_norm[3].mean():.4f} G={s2_norm[2].mean():.4f} B={s2_norm[1].mean():.4f}")

device = "cuda" if torch.cuda.is_available() else "cpu"
config = DSen2CRConfig(device=device)
model = build_model(config).to(device)
model.eval()
load_checkpoint(pathlib.Path("checkpoints/best_model.pth"), model, device=device)

H, W = s2_norm.shape[1], s2_norm.shape[2]
th, tw = min(H, 256), min(W, 256)
s2t = torch.from_numpy(s2_norm[:, :th, :tw].astype(np.float32)).unsqueeze(0).to(device)
s1t = torch.from_numpy(s1_norm[:, :th, :tw].astype(np.float32)).unsqueeze(0).to(device)

with torch.inference_mode():
    pred_norm = model(s2t, s1t).detach().cpu().float().numpy()[0]  # (13, H, W)

print(f"\n  Model raw output (normalized space):")
print(f"  shape={pred_norm.shape}")
print(f"  range=[{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
print(f"  RGB norm: R={pred_norm[3].mean():.4f} G={pred_norm[2].mean():.4f} B={pred_norm[1].mean():.4f}")
print(f"  Input vs output shift (RGB):")
s2t_np = s2t.cpu().numpy()[0]
for idx, name in [(3,"B4/R"),(2,"B3/G"),(1,"B2/B")]:
    inp = s2t_np[idx].mean()
    out = pred_norm[idx].mean()
    print(f"    {name}: in={inp:.4f}  out={out:.4f}  shift={out-inp:+.4f}")

pred_denorm = denormalize_s2(pred_norm, norm_stats)
pred_denorm = np.clip(pred_denorm, 0.0, None).astype(np.float32)

print(f"\n  Denormalized output (physical space):")
print(f"  range=[{pred_denorm.min():.1f}, {pred_denorm.max():.1f}]")
print(f"  {'Band':<6} {'Mean':>10} {'Max':>10}")
for i,bn in enumerate(s2_band_names):
    b = pred_denorm[i]
    print(f"  {bn:<6} {b.mean():>10.2f} {b.max():>10.2f}")

orig_patch = s2_aligned[:, :th, :tw]
print(f"\n  Comparison: original vs reconstructed (RGB bands):")
print(f"  Band  | OrigMean | ReconMean | Ratio")
for idx, name in [(3,"B4/R"),(2,"B3/G"),(1,"B2/B")]:
    om = float(np.nanmean(orig_patch[idx]))
    rm = float(pred_denorm[idx].mean())
    print(f"  {name:<6} | {om:>8.1f} | {rm:>9.1f} | {rm/max(om,1e-6):.3f}x")

# ---- STEP 4: Percentile stretch values ----
print(f"\n{SEP}\nSTEP 4: PERCENTILE STRETCH DIAGNOSTICS\n{SEP}")
orig_rgb = orig_patch[[3,2,1]]
pred_rgb = pred_denorm[[3,2,1]]

orig_finite = orig_rgb[np.isfinite(orig_rgb)]
pred_finite = pred_rgb[np.isfinite(pred_rgb)]

if orig_finite.size > 0 and pred_finite.size > 0:
    o2, o98 = np.percentile(orig_finite, [2, 98])
    p2, p98 = np.percentile(pred_finite, [2, 98])
    print(f"  Original  [2%,98%] stretch: {o2:.1f} - {o98:.1f}  range={o98-o2:.1f}")
    print(f"  Predicted [2%,98%] stretch: {p2:.1f} - {p98:.1f}  range={p98-p2:.1f}")

    # What color would a mean-reflectance pixel be in each image?
    orig_mid = np.nanmean(orig_finite)
    pred_mid = pred_finite.mean()
    orig_scaled = np.clip((orig_mid - o2) / (o98 - o2), 0, 1)
    pred_scaled = np.clip((pred_mid - p2) / (p98 - p2), 0, 1)
    print(f"  Scaled value of mean pixel: orig={orig_scaled:.3f}  recon={pred_scaled:.3f}")
    print(f"  STRETCH IS INDEPENDENT for each image (correct for fair display)")

# ---- STEP 5: Render comparison images ----
print(f"\n{SEP}\nSTEP 5: RENDERING COMPARISON IMAGES\n{SEP}")
from cloudremoval.evaluation.visualizer import to_rgb_numpy, reconstruction_to_rgb_numpy, RGB_INDICES

print(f"  RGB_INDICES = {RGB_INDICES}")
print(f"  to_rgb_numpy uses:              shared percentile stretch across R+G+B")
print(f"  reconstruction_to_rgb_numpy:    gray-world + median filter + desaturation")

# Original with to_rgb_numpy
rgb_orig = to_rgb_numpy(orig_patch, rgb_indices=RGB_INDICES)

# Reconstructed with to_rgb_numpy (IDENTICAL to original)
rgb_recon_same = to_rgb_numpy(pred_denorm, rgb_indices=RGB_INDICES)

# Reconstructed with reconstruction_to_rgb_numpy
rgb_recon_gw   = reconstruction_to_rgb_numpy(pred_denorm, rgb_indices=RGB_INDICES)

print(f"\n  original to_rgb_numpy:   R={rgb_orig[...,0].mean():.4f} G={rgb_orig[...,1].mean():.4f} B={rgb_orig[...,2].mean():.4f}")
print(f"  recon to_rgb_numpy:      R={rgb_recon_same[...,0].mean():.4f} G={rgb_recon_same[...,1].mean():.4f} B={rgb_recon_same[...,2].mean():.4f}")
print(f"  recon reconstruction_to_rgb: R={rgb_recon_gw[...,0].mean():.4f} G={rgb_recon_gw[...,1].mean():.4f} B={rgb_recon_gw[...,2].mean():.4f}")

# Detect color cast
r_dominates = rgb_recon_same[...,0].mean() > rgb_recon_same[...,1].mean() * 1.1
g_dominates = rgb_recon_same[...,1].mean() > rgb_recon_same[...,0].mean() * 1.1
b_dominates = rgb_recon_same[...,2].mean() > rgb_recon_same[...,0].mean() * 1.1
print(f"\n  Color cast in recon (same renderer):")
print(f"    R dominant: {r_dominates}")
print(f"    G dominant: {g_dominates}")
print(f"    B dominant: {b_dominates}")

# ---- STEP 6: Key issue check ----
print(f"\n{SEP}\nSTEP 6: ISSUE IDENTIFICATION\n{SEP}")
print(f"\n  Check: Is there a dynamic range mismatch?")
print(f"  Original B4 mean: {np.nanmean(orig_patch[3]):.1f} (expected: raw S2 TOA ~0-10000)")
print(f"  Recon    B4 mean: {pred_denorm[3].mean():.1f}")

# Critical check: is the output in the right PHYSICAL range?
if pred_denorm.max() < 100:
    print(f"\n  *** CRITICAL BUG FOUND: Model output is in wrong scale! ***")
    print(f"  Expected max ~3000-10000, got {pred_denorm.max():.1f}")
    print(f"  ROOT CAUSE: Model output NOT properly denormalized OR wrong scale")
elif pred_denorm.max() < 500:
    print(f"\n  *** WARNING: Suspiciously low range {pred_denorm.max():.1f} ***")
else:
    print(f"\n  Range looks OK: max={pred_denorm.max():.1f}")
    print(f"  The dynamic range is correct (0-{pred_denorm.max():.0f})")

# Compare band ratios (vegetation/cloud detection)
B3_mean = float(np.nanmean(orig_patch[2]))  # Green
B4_mean = float(np.nanmean(orig_patch[3]))  # Red
print(f"\n  Original B4/B3 ratio: {B4_mean/max(B3_mean,1):.3f} (>1 = red-heavy, typical for clouds/bright)")
B3_r = pred_denorm[2].mean()
B4_r = pred_denorm[3].mean()
print(f"  Recon    B4/B3 ratio: {B4_r/max(B3_r,1):.3f}")

print(f"\n  Original B2/B4 ratio (blue/red): {float(np.nanmean(orig_patch[1]))/max(float(np.nanmean(orig_patch[3])),1):.3f}")
print(f"  Recon    B2/B4 ratio (blue/red): {pred_denorm[1].mean()/max(pred_denorm[3].mean(),1):.3f}")

print(f"\n  If B2 >> B4 in recon but B2 ~= B4 in original:")
print(f"  --> color cast towards BLUE/CYAN (gray sky look)")
print(f"  If B4 >> B3 >> B2 in recon but similar in original:")
print(f"  --> color cast towards RED/PINK (burned look)")

# ---- STEP 7: Save images ----
out_dir = pathlib.Path("outputs/rgb_debug")
out_dir.mkdir(parents=True, exist_ok=True)

plt.imsave(out_dir / "A_original_to_rgb_numpy.png", rgb_orig)
plt.imsave(out_dir / "B_recon_to_rgb_numpy_same.png", rgb_recon_same)
plt.imsave(out_dir / "C_recon_reconstruction_fn.png", rgb_recon_gw)

# Big comparison
fig, ax = plt.subplots(1, 3, figsize=(18, 7), dpi=150)
fig.patch.set_facecolor("#181818")
titles = ["A: Original (to_rgb_numpy)",
          "B: Recon (to_rgb_numpy - SAME fn)",
          "C: Recon (reconstruction_to_rgb_numpy)"]
imgs   = [rgb_orig, rgb_recon_same, rgb_recon_gw]
clrs   = ["white", "#00ffcc", "yellow"]
for i,(img,t,c) in enumerate(zip(imgs,titles,clrs)):
    ax[i].imshow(img)
    ax[i].set_title(t, color=c, fontsize=10)
    ax[i].axis("off")
plt.suptitle("RGB Diagnostic - comparing renderers", color="white", fontsize=12)
plt.tight_layout()
plt.savefig(out_dir / "comparison.png", bbox_inches="tight", facecolor="#181818")
plt.close()
print(f"\n  Saved to: {out_dir}/")

print(f"\n{'='*60}\nRGB DIAGNOSTIC COMPLETE\n{'='*60}")
print(f"  Original RGB R/G/B:         {rgb_orig[...,0].mean():.4f} / {rgb_orig[...,1].mean():.4f} / {rgb_orig[...,2].mean():.4f}")
print(f"  Reconstructed (same fn):    {rgb_recon_same[...,0].mean():.4f} / {rgb_recon_same[...,1].mean():.4f} / {rgb_recon_same[...,2].mean():.4f}")
print(f"  Reconstructed (diff fn):    {rgb_recon_gw[...,0].mean():.4f} / {rgb_recon_gw[...,1].mean():.4f} / {rgb_recon_gw[...,2].mean():.4f}")
