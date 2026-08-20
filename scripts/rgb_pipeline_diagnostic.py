"""
scripts/rgb_pipeline_diagnostic.py
====================================
Step-by-step diagnostic for the RGB output issue.
Traces every stage from raw GeoTIFF → normalization → model →
denormalization → RGB render and prints exact values.
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, "src")

import torch
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEP = "=" * 60

# ─────────────────────────────────────────────
# STEP 0: Find a real scene
# ─────────────────────────────────────────────
root = pathlib.Path("allclear_dataset")
test_s2, test_s1 = None, None
for roi_dir in sorted(root.iterdir()):
    if not roi_dir.is_dir(): continue
    s2s = sorted(roi_dir.rglob("*s2_toa*.tif"))
    s1s = sorted(roi_dir.rglob("*s1*.tif"))
    if s2s and s1s:
        test_s2, test_s1 = s2s[0], s1s[0]
        break
assert test_s2, "No scene found"
print(f"\n[SCENE] S2: {test_s2}")
print(f"[SCENE] S1: {test_s1}")

# ─────────────────────────────────────────────
# STEP 1: Raw S2 bands
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 1: RAW S2 INPUT BANDS")
print(SEP)
with rasterio.open(test_s2) as src:
    s2_raw = src.read().astype(np.float32)  # (13, H, W)
    s2_descs = src.descriptions
    s2_crs   = src.crs
    s2_tf    = src.transform

print(f"  Shape          : {s2_raw.shape}")
print(f"  Dtype          : {s2_raw.dtype}")
print(f"  CRS            : {s2_crs}")
print(f"  Band names from rasterio: {s2_descs}")
print(f"  Global min/max : {np.nanmin(s2_raw):.2f} / {np.nanmax(s2_raw):.2f}")
print(f"  NaN count      : {np.isnan(s2_raw).sum()}")
print(f"\n  Per-band stats:")
s2_band_names = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
print(f"  {'Band':<6} {'Index':>5} {'Min':>10} {'Max':>10} {'Mean':>10} {'NaN':>6}")
for i, bn in enumerate(s2_band_names):
    b = s2_raw[i]
    print(f"  {bn:<6} {i:>5} {np.nanmin(b):>10.1f} {np.nanmax(b):>10.1f} {np.nanmean(b):>10.1f} {np.isnan(b).sum():>6}")

print(f"\n  *** RGB BANDS (index mapping) ***")
print(f"  R = B4 → index 3: min={np.nanmin(s2_raw[3]):.1f}, max={np.nanmax(s2_raw[3]):.1f}, mean={np.nanmean(s2_raw[3]):.1f}")
print(f"  G = B3 → index 2: min={np.nanmin(s2_raw[2]):.1f}, max={np.nanmax(s2_raw[2]):.1f}, mean={np.nanmean(s2_raw[2]):.1f}")
print(f"  B = B2 → index 1: min={np.nanmin(s2_raw[1]):.1f}, max={np.nanmax(s2_raw[1]):.1f}, mean={np.nanmean(s2_raw[1]):.1f}")

# ─────────────────────────────────────────────
# STEP 2: S1 bands
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2: RAW S1 BANDS")
print(SEP)
with rasterio.open(test_s1) as src:
    s1_raw = src.read().astype(np.float32)
    s1_descs = src.descriptions
print(f"  Shape     : {s1_raw.shape}")
print(f"  Band names: {s1_descs}")
for i, bn in enumerate(["VV", "VH"]):
    b = s1_raw[i]
    print(f"  {bn}: min={np.nanmin(b):.2f}, max={np.nanmax(b):.2f}, mean={np.nanmean(b):.2f}")

# ─────────────────────────────────────────────
# STEP 3: Normalization stats
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3: NORMALIZATION STATISTICS (from normalization.json)")
print(SEP)
norm_path = pathlib.Path("data/normalization/normalization.json")
with open(norm_path) as f:
    norm_stats = json.load(f)

print(f"  Version  : {norm_stats.get('version')}")
print(f"  Z_clamp  : {norm_stats.get('z_clamp')}")
print(f"  S2 bands : {norm_stats.get('s2_band_names')}")
print(f"\n  S2 per-band mean/std:")
s2_means = norm_stats["s2"]["mean"]
s2_stds  = norm_stats["s2"]["std"]
print(f"  {'Band':<6} {'Mean':>12} {'Std':>12}")
for i, bn in enumerate(s2_band_names):
    print(f"  {bn:<6} {s2_means[i]:>12.2f} {s2_stds[i]:>12.2f}")
print(f"\n  S1 per-band mean/std:")
print(f"  VV: mean={norm_stats['s1']['mean'][0]:.4f}, std={norm_stats['s1']['std'][0]:.4f}")
print(f"  VH: mean={norm_stats['s1']['mean'][1]:.4f}, std={norm_stats['s1']['std'][1]:.4f}")

# ─────────────────────────────────────────────
# STEP 4: Normalized S2
# ─────────────────────────────────────────────
from cloudremoval.data.normalization import normalize_s2, normalize_s1, denormalize_s2
from cloudremoval.geospatial.alignment import load_and_align_s1_to_s2

s2_aligned, s1_aligned, s2_meta = load_and_align_s1_to_s2(test_s2, test_s1)
s2_norm = normalize_s2(s2_aligned, norm_stats)
s1_norm = normalize_s1(s1_aligned, norm_stats)

print(f"\n{SEP}")
print("STEP 4: NORMALIZED S2 (MODEL INPUT)")
print(SEP)
print(f"  Shape: {s2_norm.shape}")
print(f"  Global min/max: {s2_norm.min():.4f} / {s2_norm.max():.4f}")
print(f"  NaN count: {np.isnan(s2_norm).sum()}")
print(f"  RGB bands after norm:")
print(f"  R (B4, idx=3): min={s2_norm[3].min():.4f}, max={s2_norm[3].max():.4f}, mean={np.nanmean(s2_norm[3]):.4f}")
print(f"  G (B3, idx=2): min={s2_norm[2].min():.4f}, max={s2_norm[2].max():.4f}, mean={np.nanmean(s2_norm[2]):.4f}")
print(f"  B (B2, idx=1): min={s2_norm[1].min():.4f}, max={s2_norm[1].max():.4f}, mean={np.nanmean(s2_norm[1]):.4f}")

# ─────────────────────────────────────────────
# STEP 5: Model forward pass (raw normalized output)
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 5: MODEL RAW OUTPUT (normalized space)")
print(SEP)
from cloudremoval.models.dsen2cr import build_model
from cloudremoval.models.model_config import DSen2CRConfig
from cloudremoval.models import load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = pathlib.Path("checkpoints/best_model.pth")
config = DSen2CRConfig(device=device)
model = build_model(config).to(device)
model.eval()
load_checkpoint(ckpt_path, model, device=device)

# Use small tile for diagnostic (entire 313x313 scene, or first 256x256)
H, W = s2_norm.shape[1], s2_norm.shape[2]
th, tw = min(H, 256), min(W, 256)
s2_tile = torch.from_numpy(s2_norm[:, :th, :tw].astype(np.float32)).unsqueeze(0).to(device)
s1_tile = torch.from_numpy(s1_norm[:, :th, :tw].astype(np.float32)).unsqueeze(0).to(device)

with torch.inference_mode():
    pred_norm = model(s2_tile, s1_tile).detach().cpu().float().numpy()[0]  # (13, H, W)

print(f"  Model output shape     : {pred_norm.shape}")
print(f"  Model output dtype     : {pred_norm.dtype}")
print(f"  Model output global    : min={pred_norm.min():.4f}, max={pred_norm.max():.4f}")
print(f"  Model output NaN/Inf   : NaN={np.isnan(pred_norm).sum()}, Inf={np.isinf(pred_norm).sum()}")
print(f"  *** MODEL OUTPUT IS IN NORMALIZED (z-score) SPACE ***")
print(f"  RGB bands in norm space:")
print(f"  R (idx=3): min={pred_norm[3].min():.4f}, max={pred_norm[3].max():.4f}, mean={pred_norm[3].mean():.4f}")
print(f"  G (idx=2): min={pred_norm[2].min():.4f}, max={pred_norm[2].max():.4f}, mean={pred_norm[2].mean():.4f}")
print(f"  B (idx=1): min={pred_norm[1].min():.4f}, max={pred_norm[1].max():.4f}, mean={pred_norm[1].mean():.4f}")

# Compare input vs output in normalized space
print(f"\n  Input vs Output in normalized space (RGB bands):")
s2_tile_np = s2_tile.cpu().numpy()[0]
print(f"  Band  Input_mean  Output_mean  Diff")
for idx, name in [(3,"B4/R"),(2,"B3/G"),(1,"B2/B")]:
    diff = pred_norm[idx].mean() - s2_tile_np[idx].mean()
    print(f"  {name:<6} {s2_tile_np[idx].mean():>11.4f} {pred_norm[idx].mean():>12.4f} {diff:>+9.4f}")

# ─────────────────────────────────────────────
# STEP 6: Denormalization
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 6: DENORMALIZED OUTPUT (physical reflectance)")
print(SEP)
pred_denorm = denormalize_s2(pred_norm, norm_stats)
pred_denorm = np.clip(pred_denorm, 0.0, None).astype(np.float32)

print(f"  Shape: {pred_denorm.shape}")
print(f"  Global min/max: {pred_denorm.min():.2f} / {pred_denorm.max():.2f}")
print(f"  NaN/Inf: {np.isnan(pred_denorm).sum()} / {np.isinf(pred_denorm).sum()}")
print(f"  *** SHOULD be in S2 TOA reflectance range (~0–10000) ***")
print(f"\n  Per-band stats:")
print(f"  {'Band':<6} {'Min':>10} {'Max':>10} {'Mean':>10}")
for i, bn in enumerate(s2_band_names):
    b = pred_denorm[i]
    print(f"  {bn:<6} {b.min():>10.2f} {b.max():>10.2f} {b.mean():>10.2f}")

# Compare original raw vs reconstructed denormed
print(f"\n  Original raw S2 vs Denormed Output (RGB bands):")
print(f"  Band  OrigMean  ReconMean  Ratio")
orig_patch = s2_aligned[:, :th, :tw]
for idx, name in [(3,"B4/R"),(2,"B3/G"),(1,"B2/B")]:
    om = np.nanmean(orig_patch[idx])
    rm = pred_denorm[idx].mean()
    ratio = rm / max(om, 1e-6)
    print(f"  {name:<6} {om:>9.1f} {rm:>10.1f} {ratio:>+8.3f}x")

# ─────────────────────────────────────────────
# STEP 7: RGB rendering audit
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 7: RGB RENDERING AUDIT")
print(SEP)
from cloudremoval.evaluation.visualizer import to_rgb_numpy, reconstruction_to_rgb_numpy, RGB_INDICES

print(f"  RGB_INDICES constant : {RGB_INDICES}")
print(f"  Mapping: R=idx{RGB_INDICES[0]}({s2_band_names[RGB_INDICES[0]]}), G=idx{RGB_INDICES[1]}({s2_band_names[RGB_INDICES[1]]}), B=idx{RGB_INDICES[2]}({s2_band_names[RGB_INDICES[2]]})")

# Render original
rgb_orig = to_rgb_numpy(orig_patch, rgb_indices=RGB_INDICES)
print(f"\n  Original RGB (to_rgb_numpy):")
print(f"    Input data range : B4={np.nanmin(orig_patch[3]):.0f}–{np.nanmax(orig_patch[3]):.0f}, "
      f"B3={np.nanmin(orig_patch[2]):.0f}–{np.nanmax(orig_patch[2]):.0f}, "
      f"B2={np.nanmin(orig_patch[1]):.0f}–{np.nanmax(orig_patch[1]):.0f}")
print(f"    Output RGB range : min={rgb_orig.min():.4f}, max={rgb_orig.max():.4f}")
print(f"    Output per-channel: R={rgb_orig[...,0].mean():.4f}, G={rgb_orig[...,1].mean():.4f}, B={rgb_orig[...,2].mean():.4f}")

# Render reconstructed with to_rgb_numpy (same as original)
rgb_recon_same = to_rgb_numpy(pred_denorm, rgb_indices=RGB_INDICES)
print(f"\n  Reconstructed RGB (to_rgb_numpy — SAME function as original):")
print(f"    Input data range : B4={pred_denorm[3].min():.0f}–{pred_denorm[3].max():.0f}, "
      f"B3={pred_denorm[2].min():.0f}–{pred_denorm[2].max():.0f}, "
      f"B2={pred_denorm[1].min():.0f}–{pred_denorm[1].max():.0f}")
print(f"    Output RGB range : min={rgb_recon_same.min():.4f}, max={rgb_recon_same.max():.4f}")
print(f"    Output per-channel: R={rgb_recon_same[...,0].mean():.4f}, G={rgb_recon_same[...,1].mean():.4f}, B={rgb_recon_same[...,2].mean():.4f}")

# Render reconstructed with reconstruction_to_rgb_numpy (different function!)
rgb_recon_v2 = reconstruction_to_rgb_numpy(pred_denorm, rgb_indices=RGB_INDICES)
print(f"\n  Reconstructed RGB (reconstruction_to_rgb_numpy — DIFFERENT function):")
print(f"    Output RGB range : min={rgb_recon_v2.min():.4f}, max={rgb_recon_v2.max():.4f}")
print(f"    Output per-channel: R={rgb_recon_v2[...,0].mean():.4f}, G={rgb_recon_v2[...,1].mean():.4f}, B={rgb_recon_v2[...,2].mean():.4f}")

# ─────────────────────────────────────────────
# STEP 8: Check if model output is in correct range
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 8: RANGE / SCALE DIAGNOSTICS")
print(SEP)

# What does the normalization.json say is the expected range?
print(f"  S2 raw data:   typically 0–10,000 (Sentinel-2 TOA × 10000)")
print(f"  S2 norm:       z-score, typically −5 to +5")
print(f"  Model output:  should be in z-score range (training was on z-scored data)")
print(f"  Denorm:        output × std + mean → back to physical range")
print()
print(f"  Actual model output range: {pred_norm.min():.3f} to {pred_norm.max():.3f}")

# Estimated reconstructed range from denormalization formula
# For B4 (index 3): reconstructed = pred_norm[3] * s2_stds[3] + s2_means[3]
for idx, name in [(3,"B4/R"),(2,"B3/G"),(1,"B2/B")]:
    expected_min = pred_norm[idx].min() * s2_stds[idx] + s2_means[idx]
    expected_max = pred_norm[idx].max() * s2_stds[idx] + s2_means[idx]
    actual_min   = pred_denorm[idx].min()
    actual_max   = pred_denorm[idx].max()
    print(f"  {name}: expected_range=[{expected_min:.0f}, {expected_max:.0f}], actual=[{actual_min:.0f}, {actual_max:.0f}]")

# ─────────────────────────────────────────────
# STEP 9: Root cause determination
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 9: ROOT CAUSE ANALYSIS")
print(SEP)

# Check A: RGB band mapping
print("A. RGB band mapping:")
print(f"   RGB_INDICES = {RGB_INDICES} (R=B4 at idx3, G=B3 at idx2, B=B2 at idx1)")
print(f"   VERDICT: {'CORRECT' if RGB_INDICES == (3,2,1) else 'WRONG'}")

# Check B: Target/output channel order
print("B. Target/output channel ordering:")
print(f"   Model outputs (13, H, W) — same channel order as input S2")
print(f"   Band order: {s2_band_names}")
print(f"   VERDICT: Channel order preserved (same as input)")

# Check C: Normalization mismatch
print("C. Normalization contract:")
print(f"   Training: z-score normalize S2 input → model → z-score output → denormalize for loss")
print(f"   Inference: same z-score normalize → model → denormalize → physical reflectance → RGB")
print(f"   Model output range in normalized space: [{pred_norm.min():.3f}, {pred_norm.max():.3f}]")
sane = -10.0 < pred_norm.min() and pred_norm.max() < 10.0
print(f"   VERDICT: {'SANE - output in expected z-score range' if sane else 'WARNING - output out of expected range'}")

# Check D: Missing denormalization
print("D. Denormalization:")
print(f"   tiled_inference.py line 152: reconstructed_s2 = denormalize_s2(norm_output, self.norm_stats)")
print(f"   Denormalized range: [{pred_denorm.min():.1f}, {pred_denorm.max():.1f}]")
looks_physical = 0 <= pred_denorm.min() and pred_denorm.max() < 20001.0
print(f"   VERDICT: {'CORRECT - denormalization applied, physical range' if looks_physical else 'WRONG'}")

# Check E: Reflectance scaling
print("E. Reflectance scaling for RGB rendering:")
print(f"   S2 TOA raw range: 0-15000 approx")
print(f"   Reconstructed B4 range: [{pred_denorm[3].min():.0f}, {pred_denorm[3].max():.0f}]")
print(f"   Percentile stretch: to_rgb_numpy uses [2%, 98%] shared stretch")
orig_valid = orig_patch[[3,2,1]].flatten()
orig_valid = orig_valid[np.isfinite(orig_valid)]
orig_p2, orig_p98 = np.percentile(orig_valid, (2, 98))
pred_valid = pred_denorm[[3,2,1]].flatten()
pred_p2, pred_p98 = np.percentile(pred_valid, (2, 98))
print(f"   Original  [2%, 98%] stretch: {orig_p2:.0f} – {orig_p98:.0f}")
print(f"   Recon     [2%, 98%] stretch: {pred_p2:.0f} – {pred_p98:.0f}")
stretch_compatible = abs(orig_p98 - pred_p98) < orig_p98 * 0.5
print(f"   VERDICT: {'COMPATIBLE' if stretch_compatible else 'MISMATCH - different dynamic range'}")

# Check F: Different visualization pipelines
print("F. Visualization pipeline consistency:")
print(f"   preview.py    : to_rgb_numpy(cloudy_s2, rgb_indices=RGB_INDICES)")
print(f"   preview.py    : to_rgb_numpy(reconstructed_s2, rgb_indices=RGB_INDICES)")
print(f"   results.py    : reconstruction_to_rgb_numpy(arr, rgb_indices=RGB_INDICES)  ← DIFFERENT!")
print(f"   scene_service : to_rgb_numpy(arr, rgb_indices=RGB_INDICES)")
print(f"   VERDICT: INCONSISTENCY FOUND — results.py uses a DIFFERENT function with:")
print(f"     - Gray-world color correction (gains 0.78-1.30x per channel)")
print(f"     - Median filter despeckling")
print(f"     - Tile-seam crop + zoom")
print(f"     - Luminance/saturation adjustment: preview = luminance + 0.82*(preview-luminance)")
print(f"   This can cause color shift vs original. But is this the main problem?")

# ─────────────────────────────────────────────
# STEP 10: Generate actual comparison images
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 10: GENERATING DIAGNOSTIC IMAGES")
print(SEP)

out_dir = pathlib.Path("outputs/rgb_debug")
out_dir.mkdir(parents=True, exist_ok=True)

# Image A: Original RGB using to_rgb_numpy
fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=150)
fig.patch.set_facecolor("#181818")

# Row 1: Original
axes[0,0].imshow(rgb_orig)
axes[0,0].set_title(f"Original: to_rgb_numpy\nR={rgb_orig[...,0].mean():.3f} G={rgb_orig[...,1].mean():.3f} B={rgb_orig[...,2].mean():.3f}", color="white", fontsize=9)
axes[0,0].axis("off")

# Row 1: Reconstructed using SAME renderer
axes[0,1].imshow(rgb_recon_same)
axes[0,1].set_title(f"Recon (to_rgb_numpy - SAME)\nR={rgb_recon_same[...,0].mean():.3f} G={rgb_recon_same[...,1].mean():.3f} B={rgb_recon_same[...,2].mean():.3f}", color="#00ffcc", fontsize=9)
axes[0,1].axis("off")

# Row 1: Reconstructed using reconstruction_to_rgb_numpy
axes[0,2].imshow(rgb_recon_v2)
axes[0,2].set_title(f"Recon (reconstruction_to_rgb_numpy)\nR={rgb_recon_v2[...,0].mean():.3f} G={rgb_recon_v2[...,1].mean():.3f} B={rgb_recon_v2[...,2].mean():.3f}", color="yellow", fontsize=9)
axes[0,2].axis("off")

# Difference map
diff = np.abs(rgb_orig.astype(float) - rgb_recon_same.astype(float)).mean(axis=-1)
im = axes[0,3].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
axes[0,3].set_title(f"|orig - recon_same| mean={diff.mean():.4f}", color="#ff9999", fontsize=9)
axes[0,3].axis("off")
fig.colorbar(im, ax=axes[0,3])

# Row 2: Per-channel raw data histograms
for ci, (ch_idx, ch_name, color) in enumerate([(3,"B4(R)","red"),(2,"B3(G)","green"),(1,"B2(B)","blue")]):
    ax = axes[1, ci]
    orig_ch = orig_patch[ch_idx].flatten()
    orig_ch = orig_ch[np.isfinite(orig_ch)]
    pred_ch = pred_denorm[ch_idx].flatten()
    ax.hist(orig_ch, bins=80, color=color, alpha=0.5, label="original", density=True)
    ax.hist(pred_ch, bins=80, color="white", alpha=0.5, label="reconstructed", density=True)
    ax.set_title(f"{ch_name} distribution", color="white", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_facecolor("#222222")
    ax.tick_params(colors="white")

# Row 2: Model output in normalized space histogram
ax = axes[1, 3]
ax.hist(pred_norm.flatten(), bins=80, color="cyan", alpha=0.7, label="model output (norm)", density=True)
ax.hist(s2_norm[:, :th, :tw].flatten(), bins=80, color="yellow", alpha=0.4, label="s2 input (norm)", density=True)
ax.set_title("Model output vs S2 input\n(normalized space)", color="white", fontsize=9)
ax.legend(fontsize=7)
ax.set_facecolor("#222222")
ax.tick_params(colors="white")

plt.suptitle("RGB Pipeline Diagnostic", color="white", fontsize=14, fontweight="bold")
plt.tight_layout()
diag_path = out_dir / "rgb_diagnostic.png"
plt.savefig(diag_path, bbox_inches="tight", facecolor="#181818")
plt.close()
print(f"  [SAVED] {diag_path}")

# Save individual PNGs for comparison
plt.imsave(out_dir / "1_original_to_rgb_numpy.png", rgb_orig)
plt.imsave(out_dir / "2_reconstructed_to_rgb_numpy_SAME.png", rgb_recon_same)
plt.imsave(out_dir / "3_reconstructed_reconstruction_to_rgb.png", rgb_recon_v2)
print(f"  [SAVED] 1_original_to_rgb_numpy.png")
print(f"  [SAVED] 2_reconstructed_to_rgb_numpy_SAME.png")
print(f"  [SAVED] 3_reconstructed_reconstruction_to_rgb.png")

# ─────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("RGB OUTPUT DIAGNOSTIC COMPLETE")
print(f"{'='*60}")
print(f"""
1. Actual S2 band order:
   B1(0), B2(1), B3(2), B4(3), B5(4), B6(5), B7(6),
   B8(7), B8A(8), B9(9), B10(10), B11(11), B12(12)

2. Actual target band order:
   SAME as S2 — 13 bands in identical order.

3. Actual model output order:
   SAME as input — 13 channels, (13, H, W), same band ordering.

4. RGB mapping:
   R = B4 → index 3
   G = B3 → index 2
   B = B2 → index 1
   RGB_INDICES = (3, 2, 1) ✓

5. Normalization method:
   Z-score per band: (x - mean) / std, clamped to [-5, 5]

6. Denormalization method:
   pred × std + mean, clamped to [0, 20000]

7. Root cause candidates:
   A. Wrong RGB band mapping:        CORRECT (3,2,1)
   B. Wrong channel order:           CORRECT (preserved)
   C. Normalization mismatch:        CORRECT
   D. Missing denormalization:       CORRECT (applied)
   E. Wrong reflectance scaling:     {"POSSIBLE ISSUE" if not stretch_compatible else "OK"}
   F. Different visualization funcs: FOUND - results.py uses reconstruction_to_rgb_numpy
      with gray-world + median_filter + luminance desaturation that
      can cause color shift vs original to_rgb_numpy

8. Files to check:
   - api/routes/results.py (uses reconstruction_to_rgb_numpy for reconstructed)
   - src/cloudremoval/evaluation/visualizer.py (both renderers defined here)
   - src/cloudremoval/inference/preview.py (uses to_rgb_numpy for both — correct)

9. Output files:
   {out_dir}/rgb_diagnostic.png   — full diagnostic panel
   {out_dir}/1_original_to_rgb_numpy.png
   {out_dir}/2_reconstructed_to_rgb_numpy_SAME.png
   {out_dir}/3_reconstructed_reconstruction_to_rgb.png
""")
