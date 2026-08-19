"""
Tests that all cloudremoval sub-packages are importable without error.

These tests catch missing __init__.py files, circular imports, and
syntax errors before any real work is attempted.
"""

from __future__ import annotations


class TestTopLevelImport:
    def test_cloudremoval_importable(self) -> None:
        import cloudremoval  # noqa: F401

    def test_cloudremoval_version_defined(self) -> None:
        import cloudremoval
        assert hasattr(cloudremoval, "__version__")
        assert isinstance(cloudremoval.__version__, str)


class TestSubPackageImports:
    def test_config_importable(self) -> None:
        import cloudremoval.config  # noqa: F401

    def test_utils_importable(self) -> None:
        import cloudremoval.utils  # noqa: F401

    def test_data_importable(self) -> None:
        import cloudremoval.data  # noqa: F401

    def test_models_importable(self) -> None:
        import cloudremoval.models  # noqa: F401

    def test_training_importable(self) -> None:
        import cloudremoval.training  # noqa: F401

    def test_evaluation_importable(self) -> None:
        import cloudremoval.evaluation  # noqa: F401

    def test_inference_importable(self) -> None:
        import cloudremoval.inference  # noqa: F401

    def test_preprocessing_importable(self) -> None:
        import cloudremoval.preprocessing  # noqa: F401

    def test_geospatial_importable(self) -> None:
        import cloudremoval.geospatial  # noqa: F401

    def test_cloud_importable(self) -> None:
        import cloudremoval.cloud  # noqa: F401


class TestPublicApiImports:
    def test_get_settings_importable(self) -> None:
        from cloudremoval.config import get_settings  # noqa: F401

    def test_settings_class_importable(self) -> None:
        from cloudremoval.config import Settings  # noqa: F401

    def test_setup_logging_importable(self) -> None:
        from cloudremoval.utils import setup_logging  # noqa: F401


class TestThirdPartyDeps:
    """Verify key third-party packages installed correctly."""

    def test_numpy_importable(self) -> None:
        import numpy  # noqa: F401

    def test_rasterio_importable(self) -> None:
        import rasterio  # noqa: F401

    def test_pydantic_importable(self) -> None:
        import pydantic  # noqa: F401

    def test_pydantic_settings_importable(self) -> None:
        import pydantic_settings  # noqa: F401

    def test_fastapi_importable(self) -> None:
        import fastapi  # noqa: F401

    def test_sqlalchemy_importable(self) -> None:
        import sqlalchemy  # noqa: F401

    def test_tqdm_importable(self) -> None:
        import tqdm  # noqa: F401
