# PHASE 4 TRAINING REPORT

Generated: 2026-08-20T01:33:02.199045

---

## 1. Dataset

| Split | Patches |
|-------|---------|
| Train | 282 |
| Val   | 39 |

- Train manifest: `data/manifests/train.json`
- Val manifest:   `data/manifests/val.json`
- Normalization:  `data/normalization/normalization.json`

---

## 2. Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Modified DSen2-CR (early SAR concat) |
| S2 channels | 13 |
| S1 channels | 2 |
| Target channels | 13 |
| Patch size | 256×256 |
| Base features | 256 |
| Residual blocks | 16 |
| Parameters | 18,947,341 |

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Weight decay | 1e-05 |
| Scheduler | CosineAnnealingLR (T_max=100, eta_min=1e-6) |
| Loss | L1 (CloudRemovalLoss) |
| Batch size | 4 |
| Grad accumulation | 1 |
| Effective batch | 4 |
| AMP | Enabled |
| Max epochs | 100 |
| Early stopping | patience=10 |
| Seed | 42 |

---

## 4. Results

| Metric | Value |
|--------|-------|
| Epochs completed | 54 / 100 |
| Early stopping | YES |
| Best epoch | 44 |
| Best val loss | 0.181997 |
| Best PSNR | 33.2124 dB |
| Best SSIM | 0.744177 |
| Best MAE | 0.181997 |
| Best RMSE | 0.267648 |

---

## 5. Training Duration

| Metric | Value |
|--------|-------|
| Total time | 9434.3s (157.2 min) |
| Avg epoch time | 174.7s |

---

## 6. Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8.0 GB |
| Peak VRAM (training) | 4.942 GB |
| AMP | Enabled |

---

## 7. Checkpoint Locations

- Best model: `D:\allclear_test_proi1_v1\checkpoints\best_model.pth`
- Latest:     `D:\allclear_test_proi1_v1\checkpoints\latest.pth`
- All epochs: `D:\allclear_test_proi1_v1\checkpoints/epoch_NNN.pth`

---

## 8. Logs

- Log dir:     `D:\allclear_test_proi1_v1\logs\training\20260819_225508`
- Metrics CSV: `D:\allclear_test_proi1_v1\logs\training\20260819_225508/training_metrics.csv`
- Config JSON: `D:\allclear_test_proi1_v1\logs\training\20260819_225508/training_config.json`
- Metrics JSON:`D:\allclear_test_proi1_v1\logs\training\20260819_225508/training_metrics.json`

---

## 9. Resume Capability

```bash
# Resume from latest:
python scripts/train.py --resume checkpoints/latest.pth

# Extend to 70 epochs:
python scripts/train.py --resume checkpoints/best_model.pth --epochs 70
```

---

## 10. Post-Training Verification

- Verification passed: YES

---

## 11. Next Steps

- Phase 5: Test-set evaluation using `checkpoints/best_model.pth`
- DO NOT start Phase 5 automatically.
