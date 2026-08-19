"""
tests/models/test_dsen2cr.py
==============================
Pytest tests for Phase 3 DSen2-CR model.

Tests are lightweight — no training epochs, no full dataset loading.
All tensor operations use synthetic data unless specified otherwise.

Coverage:
  - DSen2CRConfig validation
  - SARFusion shape and error handling
  - DSen2CR model initialization
  - S2/S1 channel count enforcement
  - Forward pass (CPU + CUDA)
  - Output shape correctness
  - Output finite value check
  - Loss calculation
  - Backward pass + gradient existence
  - Checkpoint save + load (round-trip)
  - AMP path (CUDA only)
  - Invalid input shape handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cloudremoval.models.model_config import (
    DSen2CRConfig,
    S2_CHANNELS,
    S1_CHANNELS,
    TARGET_CHANNELS,
    PATCH_SIZE,
)
from cloudremoval.models.fusion import SARFusion
from cloudremoval.models.dsen2cr import DSen2CR, build_model
from cloudremoval.models.losses import CloudRemovalLoss, build_loss
from cloudremoval.models.checkpoint import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> DSen2CRConfig:
    """Lightweight config for testing: fewer DRBs, fewer features."""
    return DSen2CRConfig(
        base_features=32,    # small for fast tests
        num_res_blocks=4,    # small for fast tests
    )


@pytest.fixture(scope="module")
def model(config: DSen2CRConfig) -> DSen2CR:
    return build_model(config)


@pytest.fixture(scope="module")
def criterion(config: DSen2CRConfig) -> CloudRemovalLoss:
    return build_loss(config)


def _make_batch(batch_size: int = 1, device: str = "cpu") -> tuple:
    """Create synthetic (s2, s1, target) batch tensors."""
    s2     = torch.randn(batch_size, S2_CHANNELS,    PATCH_SIZE, PATCH_SIZE, device=device)
    s1     = torch.randn(batch_size, S1_CHANNELS,    PATCH_SIZE, PATCH_SIZE, device=device)
    target = torch.randn(batch_size, TARGET_CHANNELS, PATCH_SIZE, PATCH_SIZE, device=device)
    return s2, s1, target


# ---------------------------------------------------------------------------
# 1. Config tests
# ---------------------------------------------------------------------------

class TestDSen2CRConfig:

    def test_default_channels(self) -> None:
        cfg = DSen2CRConfig()
        assert cfg.s2_channels     == S2_CHANNELS
        assert cfg.s1_channels     == S1_CHANNELS
        assert cfg.target_channels == TARGET_CHANNELS
        assert cfg.patch_size      == PATCH_SIZE

    def test_fused_channels_property(self) -> None:
        cfg = DSen2CRConfig()
        assert cfg.fused_channels == S2_CHANNELS + S1_CHANNELS  # 15

    def test_validate_passes(self) -> None:
        cfg = DSen2CRConfig()
        cfg.validate()  # should not raise

    def test_validate_wrong_s2(self) -> None:
        cfg = DSen2CRConfig(s2_channels=10)  # wrong
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_validate_wrong_s1(self) -> None:
        cfg = DSen2CRConfig(s1_channels=4)  # wrong
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_to_dict_roundtrip(self) -> None:
        cfg = DSen2CRConfig()
        d = cfg.to_dict()
        restored = DSen2CRConfig.from_dict(d)
        assert restored.s2_channels     == cfg.s2_channels
        assert restored.s1_channels     == cfg.s1_channels
        assert restored.base_features   == cfg.base_features
        assert restored.num_res_blocks  == cfg.num_res_blocks


# ---------------------------------------------------------------------------
# 2. Fusion tests
# ---------------------------------------------------------------------------

class TestSARFusion:

    def test_output_shape(self, config: DSen2CRConfig) -> None:
        fusion = SARFusion(config)
        s2 = torch.randn(2, S2_CHANNELS, 64, 64)
        s1 = torch.randn(2, S1_CHANNELS,  64, 64)
        out = fusion(s2, s1)
        assert out.shape == (2, config.fused_channels, 64, 64)

    def test_wrong_s2_channels(self, config: DSen2CRConfig) -> None:
        fusion = SARFusion(config)
        s2 = torch.randn(1, 10, 64, 64)   # wrong channel count
        s1 = torch.randn(1, S1_CHANNELS, 64, 64)
        with pytest.raises(ValueError, match="S2 channels"):
            fusion(s2, s1)

    def test_wrong_s1_channels(self, config: DSen2CRConfig) -> None:
        fusion = SARFusion(config)
        s2 = torch.randn(1, S2_CHANNELS, 64, 64)
        s1 = torch.randn(1, 4, 64, 64)  # wrong
        with pytest.raises(ValueError, match="S1 channels"):
            fusion(s2, s1)

    def test_spatial_mismatch(self, config: DSen2CRConfig) -> None:
        fusion = SARFusion(config)
        s2 = torch.randn(1, S2_CHANNELS, 64, 64)
        s1 = torch.randn(1, S1_CHANNELS, 32, 32)   # wrong spatial
        with pytest.raises(ValueError, match="Spatial"):
            fusion(s2, s1)

    def test_batch_mismatch(self, config: DSen2CRConfig) -> None:
        fusion = SARFusion(config)
        s2 = torch.randn(2, S2_CHANNELS, 64, 64)
        s1 = torch.randn(1, S1_CHANNELS, 64, 64)   # wrong batch
        with pytest.raises(ValueError, match="Batch"):
            fusion(s2, s1)


# ---------------------------------------------------------------------------
# 3. Model initialization
# ---------------------------------------------------------------------------

class TestDSen2CRInit:

    def test_builds_without_error(self, config: DSen2CRConfig) -> None:
        m = build_model(config)
        assert isinstance(m, DSen2CR)

    def test_has_parameters(self, model: DSen2CR) -> None:
        assert model.parameter_count() > 0

    def test_default_config_builds(self) -> None:
        """Default config (full 256 features) must also build."""
        m = build_model()
        assert m.parameter_count() > 0

    def test_parameter_count_str(self, model: DSen2CR) -> None:
        s = model.parameter_count_str()
        assert "M" in s or "K" in s or s.isdigit()

    def test_summary_string(self, model: DSen2CR) -> None:
        s = model.summary()
        assert "DSen2-CR" in s
        assert "Parameters" in s


# ---------------------------------------------------------------------------
# 4. Forward pass (CPU)
# ---------------------------------------------------------------------------

class TestForwardPassCPU:

    def test_output_shape(self, model: DSen2CR) -> None:
        s2, s1, _ = _make_batch(1, "cpu")
        with torch.no_grad():
            out = model(s2, s1)
        assert out.shape == (1, TARGET_CHANNELS, PATCH_SIZE, PATCH_SIZE)

    def test_output_batch_size_2(self, model: DSen2CR) -> None:
        s2, s1, _ = _make_batch(2, "cpu")
        with torch.no_grad():
            out = model(s2, s1)
        assert out.shape == (2, TARGET_CHANNELS, PATCH_SIZE, PATCH_SIZE)

    def test_output_finite(self, model: DSen2CR) -> None:
        s2, s1, _ = _make_batch(1, "cpu")
        with torch.no_grad():
            out = model(s2, s1)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_invalid_s2_channels(self, model: DSen2CR) -> None:
        s2 = torch.randn(1, 10, PATCH_SIZE, PATCH_SIZE)    # wrong channels
        s1 = torch.randn(1, S1_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        with pytest.raises(ValueError):
            model(s2, s1)

    def test_invalid_s1_channels(self, model: DSen2CR) -> None:
        s2 = torch.randn(1, S2_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        s1 = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE)   # wrong channels
        with pytest.raises(ValueError):
            model(s2, s1)


# ---------------------------------------------------------------------------
# 5. Forward + backward (CPU)
# ---------------------------------------------------------------------------

class TestBackwardCPU:

    def test_loss_is_finite(self, model: DSen2CR, criterion: CloudRemovalLoss) -> None:
        s2, s1, target = _make_batch(1, "cpu")
        pred = model(s2, s1)
        loss, _ = criterion(pred, target)
        assert loss.isfinite().item()

    def test_gradients_exist(self, model: DSen2CR, criterion: CloudRemovalLoss) -> None:
        model.train()
        s2, s1, target = _make_batch(1, "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        pred = model(s2, s1)
        loss, _ = criterion(pred, target)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients found after backward"
        optimizer.zero_grad()
        model.eval()


# ---------------------------------------------------------------------------
# 6. CUDA tests (skip if no GPU)
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDA:

    def test_forward_on_cuda(self, config: DSen2CRConfig) -> None:
        m = build_model(config).cuda()
        s2, s1, _ = _make_batch(1, "cuda")
        with torch.no_grad():
            out = m(s2, s1)
        assert out.device.type == "cuda"
        assert out.shape == (1, TARGET_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        assert torch.isfinite(out).all()

    def test_amp_forward(self, config: DSen2CRConfig) -> None:
        m = build_model(config).cuda()
        c = build_loss(config).cuda()
        s2, s1, target = _make_batch(1, "cuda")
        with torch.amp.autocast(device_type="cuda"):
            pred = m(s2, s1)
            loss, _ = c(pred, target)
        assert loss.isfinite().item()

    def test_amp_backward(self, config: DSen2CRConfig) -> None:
        m = build_model(config).cuda().train()
        c = build_loss(config).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(device="cuda")
        s2, s1, target = _make_batch(1, "cuda")
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            pred = m(s2, s1)
            loss, _ = c(pred, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        assert loss.isfinite().item()


# ---------------------------------------------------------------------------
# 7. Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def test_save_and_load(self, config: DSen2CRConfig) -> None:
        model = build_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            save_checkpoint(
                checkpoint_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                epoch=5,
                val_loss=0.12345,
                train_loss=0.14567,
                config_dict=config.to_dict(),
                norm_version="v1",
                amp_enabled=False,
                early_stopping_state=None,
                is_best=True,
            )

            latest = ckpt_dir / "latest.pth"
            best   = ckpt_dir / "best_model.pth"
            assert latest.exists()
            assert best.exists()

            # Load into fresh model
            fresh = build_model(config)
            fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-4)
            ckpt = load_checkpoint(
                checkpoint_path=latest,
                model=fresh,
                optimizer=fresh_opt,
                device="cpu",
            )
            assert ckpt["epoch"] == 5
            assert abs(ckpt["val_loss"] - 0.12345) < 1e-6
            assert ckpt["norm_version"] == "v1"

            # Weights should match
            model.eval(); fresh.eval()
            s2, s1, _ = _make_batch(1, "cpu")
            with torch.no_grad():
                out_orig  = model(s2, s1)
                out_fresh = fresh(s2, s1)
            assert (out_orig - out_fresh).abs().max().item() < 1e-5

    def test_epoch_checkpoint_saved(self, config: DSen2CRConfig) -> None:
        model = build_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            save_checkpoint(
                checkpoint_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                epoch=5,
                val_loss=0.1,
                train_loss=0.1,
                config_dict=config.to_dict(),
                early_stopping_state=None,
                is_best=False,
            )
            # Phase 4: epoch checkpoint saved every epoch
            assert (ckpt_dir / "epoch_005.pth").exists()
