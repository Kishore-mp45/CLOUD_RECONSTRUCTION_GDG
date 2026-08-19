"""
Tests for the machine environment: Python version, PyTorch, and CUDA.

These tests document hardware requirements and fail loudly if the GPU
is unavailable so that configuration issues are caught early.
"""

from __future__ import annotations

import sys


class TestPythonVersion:
    def test_python_version_gte_312(self) -> None:
        assert sys.version_info >= (3, 12), (
            f"Python 3.12+ required, got {sys.version_info}"
        )


class TestPyTorch:
    def test_torch_importable(self) -> None:
        import torch  # noqa: F401

    def test_torch_version_gte_25(self) -> None:
        import torch
        major, minor, *_ = torch.__version__.split(".")
        assert (int(major), int(minor)) >= (2, 5), (
            f"PyTorch >= 2.5 required, got {torch.__version__}"
        )

    def test_torch_cuda_version_reported(self) -> None:
        import torch
        assert torch.version.cuda is not None, (
            "torch.version.cuda is None — was torch installed with CUDA support?"
        )


class TestCUDA:
    """
    CUDA tests.  If the GPU is absent these tests FAIL explicitly
    instead of silently falling back to CPU.
    """

    def test_cuda_is_available(self) -> None:
        import torch
        assert torch.cuda.is_available(), (
            "torch.cuda.is_available() returned False. "
            "Verify the CUDA-enabled PyTorch build and NVIDIA drivers."
        )

    def test_cuda_device_count_gte_1(self) -> None:
        import torch
        count = torch.cuda.device_count()
        assert count >= 1, f"Expected at least 1 CUDA device, got {count}"

    def test_cuda_get_device_name(self) -> None:
        import torch
        name = torch.cuda.get_device_name(0)
        assert isinstance(name, str) and len(name) > 0, (
            f"Unexpected GPU name: {name!r}"
        )

    def test_cuda_tensor_creation_and_operation(self) -> None:
        """Real CUDA computation test — not just is_available()."""
        import torch
        a = torch.ones(4, 4, device="cuda")
        b = torch.ones(4, 4, device="cuda")
        c = a @ b  # matrix multiply on GPU
        assert c.device.type == "cuda", "Result tensor must be on CUDA device"
        assert float(c[0, 0]) == 4.0, f"Expected 4.0, got {float(c[0, 0])}"

    def test_cuda_memory_available(self) -> None:
        import torch
        free, total = torch.cuda.mem_get_info(0)
        assert total > 0, "GPU total memory should be > 0"
        assert free >= 0, "Free GPU memory should be >= 0"


class TestLogging:
    def test_logging_setup(self, settings, tmp_path) -> None:
        from cloudremoval.config.settings import Settings
        from cloudremoval.utils.logging import setup_logging

        s = Settings(LOG_DIR=tmp_path / "logs", LOG_LEVEL="DEBUG")
        logger = setup_logging(s)
        assert logger is not None
        log_file = tmp_path / "logs" / s.LOG_FILE
        assert log_file.exists(), f"Log file not created at {log_file}"
