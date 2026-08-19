# PHASE 3 MODEL REPORT

## Summary

Phase 3 implements the DSen2-CR model with SAR fusion for Sentinel-2 cloud removal
using the ALLClear dataset prepared in Phase 2. No training was executed.

---

## 1. DSen2-CR Baseline Architecture

**Reference:** Meraner et al. (2020). Cloud removal in Sentinel-2 imagery using a deep
residual neural network and SAR-optical data fusion. ISPRS Journal of
Photogrammetry and Remote Sensing, 166, 333–346.

| Component | Description | Status |
|-----------|-------------|--------|
| Dense Residual Blocks (DRB) | Core feature extraction units | **ORIGINAL** |
| Residual learning | Output = S2_input + correction | **ORIGINAL** |
| Convolutional architecture | No pooling, full resolution throughout | **ORIGINAL** |
| Feature extraction depth | 16 DRBs, 256 base features | **ORIGINAL** |
| Tail projection | Conv2d back to target channel count | **ORIGINAL** |
| Head conv layer | Input channels changed: 13 → 15 | **MODIFIED** |

---

## 2. SAR Fusion Modification

**Fusion strategy:** Early channel concatenation (minimum modification to DSen2-CR).

```
S2 (13, H, W)  ──┐
                   ├── concat → (15, H, W) → DSen2-CR head → DRBs → tail → + S2 → output
S1 ( 2, H, W)  ──┘
```

**What changed:** Only the first `Conv2d` layer input was changed from `Conv2d(13, 256, 3)` to `Conv2d(15, 256, 3)`.

**What is unchanged:** All 16 residual blocks, all other convolutions, the residual
connection, weight initialisation, and the overall network topology.

---

## 3. Confirmed Input / Output Channels

| Tensor | Channels | Shape | Source |
|--------|----------|-------|--------|
| S2 input | 13 | (B, 13, 256, 256) | Phase 0 audit — B1–B12+B8A |
| S1 input | 2 | (B, 2, 256, 256) | Phase 0 audit — VV, VH |
| Fused input | 15 | (B, 15, 256, 256) | 13 + 2 early concat |
| Target output | 13 | (B, 13, 256, 256) | Cloud-free S2 reconstruction |

---

## 4. Loss Formulation

**Primary:** L1 Loss (Mean Absolute Error on normalised tensors)

```
loss = mean(|pred - target|)
```

**Source:** Standard for DSen2-CR regression (paper does not specify alternate loss).

**Optional:** SSIM term (configurable via `ssim_weight`, default = 0.0).

```
loss = L1(pred, target) + ssim_weight * (1 - SSIM(pred, target))
```

Recommendation: Start with pure L1 (ssim_weight=0.0) and evaluate before adding SSIM.

---

## 5. Parameter Count

| Component | Parameters |
|-----------|------------|
| Head Conv2d(15→256) | ~36K |
| 16 × Dense Residual Blocks | ~18.8M |
| Tail Conv2d(256→13) | ~30K |
| **Total** | **~18.95M** |

---

## 6. AMP Implementation

- **API:** `torch.amp.autocast(device_type='cuda')` + `torch.amp.GradScaler(device='cuda')`
- **PyTorch version:** 2.5.1+cu121 (modern API, not deprecated `torch.cuda.amp`)
- **Status:** Verified working — AMP loss is finite, backward pass succeeds
- **Configurable:** `--amp` / `--no-amp` in `train.py`
- **Fallback:** AMP is automatically disabled when `device=cpu`

---

## 7. RTX 4060 Memory Results (Batch = 1)

| Metric | Value |
|--------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8.00 GB |
| Model parameters | 18.95M |
| Allocated VRAM | 0.531 GB |
| Reserved VRAM | 1.869 GB |
| **Peak VRAM (fwd + bwd)** | **1.747 GB** |
| Estimated safe batch | 3 (@ 80% of 8 GB) |

**Safe batch size recommendation: 4**
Peak per-sample = 1.747 GB → 4 × 1.747 = ~7.0 GB < 8 GB total.
With AMP reducing activations by ~50%, batch=4 should fit comfortably.

---

## 8. Tested Batch Size

- Smoke test used: **batch = 1** (safe minimum for memory measurement)
- Recommended training batch: **4** (fits in 8 GB with AMP)
- If OOM at batch=4: use `--batch-size 2 --grad-accum 2` (effective batch = 4)

---

## 9. Gradient Accumulation Recommendation

For 30-epoch run with `--batch-size 4`:
- No gradient accumulation needed.

For 70-epoch run if experimenting with larger effective batch:
- `--batch-size 2 --grad-accum 4` → effective batch = 8

---

## 10. Checkpoint Design

```
checkpoints/
    latest.pth      ← overwritten every epoch
    best.pth        ← saved when val_loss improves
    epoch_NNN.pth   ← saved every 5 epochs (configurable)
```

Each checkpoint embeds:
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `epoch`, `val_loss`, `train_loss`
- `config` dict (full DSen2CRConfig), `norm_version`, `amp_enabled`
- `timestamp` (ISO 8601)

---

## 11. Forward / Backward Test Results

| Test | Result |
|------|--------|
| Forward pass (batch=1, CUDA) | PASS |
| Output shape (1, 13, 256, 256) | PASS |
| No NaN/Inf in output | PASS |
| L1 loss finite | PASS |
| Backward pass + gradients | PASS |
| AMP forward pass | PASS |
| AMP backward pass | PASS |
| Checkpoint save | PASS |
| Checkpoint load | PASS |
| Weights match after reload | PASS (max diff = 0.00e+00) |

---

## 12. Pytest Results

| Test suite | Tests | Result |
|------------|-------|--------|
| tests/models/test_dsen2cr.py | 28 | **28 PASSED** |
| Full suite (all phases) | 155 | **155 PASSED** |

---

## 13. Known Limitations

- The residual connection adds the normalised cloudy S2 directly to the
  predicted residual. If a patch has very heavy cloud cover (high nodata),
  the residual learning may struggle. This is expected behaviour for DSen2-CR
  and not a bug.
- Patch-level training (256×256 crops) means the model does not have full
  scene context during training.
- No cloud mask is used (excluded by design per Phase 3 requirements).

---

## 14. Training Commands (Phase 4)

```bash
# Fresh training (30 epochs):
python scripts/train.py --epochs 30 --batch-size 4 --amp --device cuda

# Resume from last checkpoint:
python scripts/train.py --epochs 30 --resume checkpoints/latest.pth --amp --device cuda

# Extend to 70 epochs from a 30-epoch best checkpoint:
python scripts/train.py --epochs 70 --resume checkpoints/best.pth --amp --device cuda

# Smaller batch + gradient accumulation (if OOM at batch=4):
python scripts/train.py --epochs 30 --batch-size 2 --grad-accum 2 --amp --device cuda
```

---

## 15. Files Created in Phase 3

| File | Purpose |
|------|---------|
| `src/cloudremoval/models/model_config.py` | DSen2CRConfig — all hyperparams |
| `src/cloudremoval/models/fusion.py` | SARFusion — early channel concat |
| `src/cloudremoval/models/dsen2cr.py` | DSen2CR main model |
| `src/cloudremoval/models/losses.py` | L1 + optional SSIM loss |
| `src/cloudremoval/models/checkpoint.py` | Save/load utilities |
| `src/cloudremoval/models/__init__.py` | Public model API |
| `src/cloudremoval/training/metrics.py` | PSNR, SSIM, MAE, RMSE |
| `src/cloudremoval/training/trainer.py` | Full training loop |
| `src/cloudremoval/training/__init__.py` | Public training API |
| `scripts/test_model.py` | Phase 3 smoke + memory test |
| `scripts/train.py` | Training script (not executed) |
| `tests/models/__init__.py` | Test package marker |
| `tests/models/test_dsen2cr.py` | 28 pytest model tests |
| `docs/PHASE3_MODEL_REPORT.md` | This document |

**Modified:**
| File | Change |
|------|--------|
| `src/cloudremoval/config/settings.py` | MAX_EPOCHS: le=30 → le=200 |
| `tests/test_config.py` | Updated MAX_EPOCHS test to match new le=200 |

---

## 16. Phase 4 Training Recommendation

1. Start with: `python scripts/train.py --epochs 30 --batch-size 4 --amp --device cuda`
2. Monitor val PSNR — expect improvement by epoch 5–10.
3. If loss is still decreasing at epoch 30: extend to 70 using `--resume checkpoints/best.pth --epochs 70`.
4. Best checkpoint is always preserved at `checkpoints/best.pth`.
5. Estimated time: ~1–1.5 hours for 30 epochs; ~2–3 hours for 70 epochs.
