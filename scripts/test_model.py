"""
scripts/test_model.py
======================
Phase 3 Model Smoke Test + GPU Memory Test

Validates the DSen2-CR model without training.

Run:
    python scripts/test_model.py

Tests:
    1.  Load Phase 2 configuration
    2.  Load one real training sample from Phase 2 manifest
    3.  Initialize DSen2-CR model
    4.  Move model + tensors to RTX 4060 (CUDA)
    5.  Forward pass
    6.  Output shape verification
    7.  Finite values check
    8.  Loss calculation
    9.  Backward pass + gradient check
    10. AMP path test
    11. Checkpoint save + load test
    12. GPU memory report

Prints [PHASE 3] prefix for all log lines as required.
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch

from cloudremoval.models import (
    DSen2CRConfig,
    DSen2CR,
    build_model,
    build_loss,
    save_checkpoint,
    load_checkpoint,
    S2_CHANNELS,
    S1_CHANNELS,
    TARGET_CHANNELS,
    PATCH_SIZE,
)

SEP = "=" * 60


def ph(msg: str) -> None:
    print(f"[PHASE 3] {msg}")


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(cond: bool, msg: str) -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


# ---------------------------------------------------------------------------
# 1. Load Phase 2 configuration
# ---------------------------------------------------------------------------
section("1 — Loading Phase 2 Configuration")

ph("Loading Phase 2 configuration...")
ph(f"S2 channels   : {S2_CHANNELS}")
ph(f"S1 channels   : {S1_CHANNELS}")
ph(f"Target channels: {TARGET_CHANNELS}")
ph(f"Patch size    : {PATCH_SIZE}")

MANIFEST_PATH = _PROJECT_ROOT / "data" / "manifests" / "train.json"
NORM_PATH     = _PROJECT_ROOT / "data" / "normalization" / "normalization.json"

check(MANIFEST_PATH.exists(), f"Train manifest exists: {MANIFEST_PATH}")
check(NORM_PATH.exists(),     f"Normalization file exists: {NORM_PATH}")

# ---------------------------------------------------------------------------
# 2. Load one real training sample
# ---------------------------------------------------------------------------
section("2 — Loading Real Phase 2 Sample")

from cloudremoval.data.dataset import build_dataset

dataset = build_dataset(MANIFEST_PATH, NORM_PATH, augment=False)
ph(f"Dataset size: {len(dataset)} patches")
check(len(dataset) > 0, "Dataset has patches")

sample = dataset[0]
s2_np     = sample["s2"]
s1_np     = sample["s1"]
target_np = sample["target"]
ph(f"s2     shape: {tuple(s2_np.shape)}")
ph(f"s1     shape: {tuple(s1_np.shape)}")
ph(f"target shape: {tuple(target_np.shape)}")

check(tuple(s2_np.shape)     == (S2_CHANNELS,     PATCH_SIZE, PATCH_SIZE), "s2 shape correct")
check(tuple(s1_np.shape)     == (S1_CHANNELS,      PATCH_SIZE, PATCH_SIZE), "s1 shape correct")
check(tuple(target_np.shape) == (TARGET_CHANNELS,  PATCH_SIZE, PATCH_SIZE), "target shape correct")

# Add batch dimension
s2_batch     = s2_np.unsqueeze(0)      # (1, 13, 256, 256)
s1_batch     = s1_np.unsqueeze(0)      # (1,  2, 256, 256)
target_batch = target_np.unsqueeze(0)  # (1, 13, 256, 256)

# ---------------------------------------------------------------------------
# 3. Build DSen2-CR model
# ---------------------------------------------------------------------------
section("3 — Building DSen2-CR Model")

ph("Building DSen2-CR...")
config = DSen2CRConfig()
model = build_model(config)
n_params = model.parameter_count()
ph(f"Parameters: {model.parameter_count_str()}")
print(model.summary())

check(n_params > 0, "Model has parameters")

# ---------------------------------------------------------------------------
# 4. CUDA test + move to GPU
# ---------------------------------------------------------------------------
section("4 — CUDA / Device")

ph("Testing CUDA...")
cuda_available = torch.cuda.is_available()
check(cuda_available, "CUDA is available")

device = torch.device("cuda" if cuda_available else "cpu")
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
ph(f"Device: {device} ({gpu_name})")

model = model.to(device)
s2_t     = s2_batch.to(device)
s1_t     = s1_batch.to(device)
target_t = target_batch.to(device)

# ---------------------------------------------------------------------------
# 5 & 6. Forward pass + output shape
# ---------------------------------------------------------------------------
section("5–6 — Forward Pass + Output Shape")

ph("Running forward pass...")
torch.cuda.synchronize()
t0 = time.time()

model.eval()
with torch.no_grad():
    output = model(s2_t, s1_t)

torch.cuda.synchronize()
elapsed_ms = (time.time() - t0) * 1000

ph(f"Output shape  : {tuple(output.shape)}")
ph(f"Forward time  : {elapsed_ms:.1f} ms")

check(
    tuple(output.shape) == (1, TARGET_CHANNELS, PATCH_SIZE, PATCH_SIZE),
    f"Output shape == (1, {TARGET_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE})"
)

# ---------------------------------------------------------------------------
# 7. Finite values
# ---------------------------------------------------------------------------
section("7 — Finite Values Check")

n_nan  = torch.isnan(output).sum().item()
n_inf  = torch.isinf(output).sum().item()
ph(f"NaN pixels : {n_nan}")
ph(f"Inf pixels : {n_inf}")
check(n_nan == 0, "Output has no NaN values")
check(n_inf == 0, "Output has no Inf values")

# ---------------------------------------------------------------------------
# 8. Loss calculation
# ---------------------------------------------------------------------------
section("8 — Loss Calculation")

ph("Running loss test...")
criterion = build_loss(config)
criterion = criterion.to(device)

model.eval()
with torch.no_grad():
    pred_out = model(s2_t, s1_t)
    loss_val, components = criterion(pred_out, target_t)

ph(f"Loss (L1)     : {components['total']:.6f}")
ph(f"Loss is finite: {loss_val.isfinite().item()}")
check(loss_val.isfinite().item(), "Loss is a finite scalar")

# ---------------------------------------------------------------------------
# 9. Backward pass + gradients
# ---------------------------------------------------------------------------
section("9 — Backward Pass + Gradient Check")

ph("Running backward test...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()

pred_train = model(s2_t, s1_t)
loss_train, _ = criterion(pred_train, target_t)
loss_train.backward()

# Check that at least one parameter received a gradient
grads_exist = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.parameters()
    if p.requires_grad
)
ph(f"Gradients exist: {grads_exist}")
check(grads_exist, "Model parameters have non-zero gradients after backward")

optimizer.zero_grad()

# ---------------------------------------------------------------------------
# 10. AMP path
# ---------------------------------------------------------------------------
section("10 — AMP (Automatic Mixed Precision) Test")

ph("Testing AMP...")
if cuda_available:
    model.train()
    scaler = torch.amp.GradScaler(device="cuda")
    optimizer.zero_grad()

    with torch.amp.autocast(device_type="cuda"):
        pred_amp = model(s2_t, s1_t)
        loss_amp, amp_components = criterion(pred_amp, target_t)

    scaler.scale(loss_amp).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    ph(f"AMP loss: {amp_components['total']:.6f}")
    ph(f"AMP loss finite: {loss_amp.isfinite().item()}")
    check(loss_amp.isfinite().item(), "AMP loss is finite")
    ph("AMP path: PASSED")
else:
    ph("AMP skipped (no CUDA)")

# ---------------------------------------------------------------------------
# 11. Checkpoint save + load
# ---------------------------------------------------------------------------
section("11 — Checkpoint Save + Load")

ph("Checkpoint test...")
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_dir = Path(tmpdir)

    save_checkpoint(
        checkpoint_dir=ckpt_dir,
        model=model,
        optimizer=optimizer,
        epoch=1,
        val_loss=loss_train.item(),
        train_loss=loss_train.item(),
        config_dict=config.to_dict(),
        norm_version="v1",
        amp_enabled=True,
        is_best=True,
    )

    latest_path = ckpt_dir / "latest.pth"
    best_path   = ckpt_dir / "best.pth"
    check(latest_path.exists(), "latest.pth was created")
    check(best_path.exists(),   "best.pth was created")

    # Reload into a fresh model
    fresh_model = build_model(config).to(device)
    fresh_opt   = torch.optim.Adam(fresh_model.parameters(), lr=1e-4)
    ckpt_data   = load_checkpoint(
        checkpoint_path=latest_path,
        model=fresh_model,
        optimizer=fresh_opt,
        device=str(device),
    )
    check(ckpt_data["epoch"] == 1, "Checkpoint epoch restored correctly")
    check(ckpt_data["norm_version"] == "v1", "Checkpoint norm_version correct")

    # Verify fresh model produces same output after weight restore
    fresh_model.eval()
    model.eval()
    with torch.no_grad():
        out_orig  = model(s2_t, s1_t)
        out_fresh = fresh_model(s2_t, s1_t)
    max_diff = (out_orig - out_fresh).abs().max().item()
    ph(f"Max weight difference after reload: {max_diff:.2e}")
    check(max_diff < 1e-5, "Reloaded model weights match original")

ph("Checkpoint test: PASSED")

# ---------------------------------------------------------------------------
# 12. GPU memory test
# ---------------------------------------------------------------------------
section("12 — GPU Memory Test")

ph("GPU memory test...")
if cuda_available:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # Re-run forward + backward for memory measurement
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast(device_type="cuda"):
        pred_mem = model(s2_t, s1_t)
        loss_mem, _ = criterion(pred_mem, target_t)

    loss_mem.backward()
    optimizer.zero_grad()

    alloc_gb   = torch.cuda.memory_allocated(0) / 1024 ** 3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024 ** 3
    peak_gb    = torch.cuda.max_memory_allocated(0) / 1024 ** 3

    total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    ph(f"GPU                 : {torch.cuda.get_device_name(0)}")
    ph(f"GPU total memory    : {total_gpu_gb:.2f} GB")
    ph(f"Model parameters    : {model.parameter_count_str()}")
    ph(f"Input batch size    : 1")
    ph(f"Input shape (s2)    : {tuple(s2_t.shape)}")
    ph(f"Input shape (s1)    : {tuple(s1_t.shape)}")
    ph(f"Allocated VRAM      : {alloc_gb:.3f} GB")
    ph(f"Reserved VRAM       : {reserved_gb:.3f} GB")
    ph(f"Peak VRAM (fwd+bwd) : {peak_gb:.3f} GB")

    # Estimate safe batch size
    safe_gb = total_gpu_gb * 0.80   # leave 20% headroom
    overhead_per_sample_gb = peak_gb  # batch=1 peak
    estimated_safe_batch = max(1, int(safe_gb / overhead_per_sample_gb))
    ph(f"Estimated safe batch: {estimated_safe_batch} "
       f"(80% of {total_gpu_gb:.1f}GB GPU)")
else:
    ph("GPU memory test skipped (no CUDA)")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  PHASE 3 COMPLETE")
print(f"{'='*60}")

ph(f"Model architecture  : DSen2-CR (Modified for SAR fusion)")
ph(f"SAR fusion method   : early_concat (minimum modification)")
ph(f"S2 channels         : {S2_CHANNELS}")
ph(f"S1 channels         : {S1_CHANNELS}")
ph(f"Target channels     : {TARGET_CHANNELS}")
ph(f"Patch size          : {PATCH_SIZE}x{PATCH_SIZE}")
ph(f"Parameters          : {model.parameter_count_str()}")
ph(f"Loss function       : L1 (original DSen2-CR)")
ph(f"AMP                 : {cuda_available}")
if cuda_available:
    ph(f"Peak VRAM (batch=1) : {peak_gb:.3f} GB")
    ph(f"Safe batch estimate : {estimated_safe_batch}")
ph("Forward pass        : PASSED")
ph("Backward pass       : PASSED")
ph("Checkpoint test     : PASSED")
ph("Tests complete.")

print(f"\n{'='*60}")
print("  Training command (run manually when ready):")
print()
print("  # Fresh training (30 epochs):")
print("  python scripts/train.py --epochs 30 --batch-size 4 --amp --device cuda")
print()
print("  # Resume from latest checkpoint:")
print("  python scripts/train.py --epochs 30 --resume checkpoints/latest.pth --amp --device cuda")
print()
print("  # Extend to 70 epochs:")
print("  python scripts/train.py --epochs 70 --resume checkpoints/best.pth --amp --device cuda")
print(f"{'='*60}")
