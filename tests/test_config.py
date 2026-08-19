"""
Tests for the configuration system (cloudremoval.config.settings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloudremoval.config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Verify default values are correct and MAX_EPOCHS is locked at 30."""

    def test_max_epochs_default(self, settings: Settings) -> None:
        assert settings.MAX_EPOCHS == 30, "MAX_EPOCHS must default to 30"

    def test_device_default(self, settings: Settings) -> None:
        assert settings.DEVICE in {"cuda", "cpu"}

    def test_checkpoint_dir_default(self, settings: Settings) -> None:
        assert isinstance(settings.CHECKPOINT_DIR, Path)

    def test_output_dir_default(self, settings: Settings) -> None:
        assert isinstance(settings.OUTPUT_DIR, Path)

    def test_log_dir_default(self, settings: Settings) -> None:
        assert isinstance(settings.LOG_DIR, Path)

    def test_log_level_default(self, settings: Settings) -> None:
        assert settings.LOG_LEVEL in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_missing_s1_strategy_default(self, settings: Settings) -> None:
        assert settings.MISSING_S1_STRATEGY in {"skip", "zeros"}


class TestSettingsValidation:
    """Verify that invalid values are rejected."""

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(DEVICE="tpu")

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(LOG_LEVEL="VERBOSE")

    def test_max_epochs_above_200_raises(self) -> None:
        """MAX_EPOCHS is capped at 200 (relaxed from 30 in Phase 3 to allow 70-epoch runs)."""
        with pytest.raises(Exception):
            Settings(MAX_EPOCHS=201)

    def test_max_epochs_70_is_valid(self) -> None:
        """70 epochs must be valid (user plans to extend from 30 to 70)."""
        s = Settings(MAX_EPOCHS=70)
        assert s.MAX_EPOCHS == 70

    def test_max_epochs_zero_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(MAX_EPOCHS=0)


class TestSettingsProperties:
    """Verify computed properties return correct Path objects."""

    def test_metadata_path_is_path(self, settings: Settings) -> None:
        assert isinstance(settings.metadata_path, Path)
        assert settings.metadata_path.name == settings.METADATA_FILE

    def test_allclear_dataset_path_is_path(self, settings: Settings) -> None:
        assert isinstance(settings.allclear_dataset_path, Path)
        assert settings.allclear_dataset_path.name == settings.ALLCLEAR_SUBDIR

    def test_best_checkpoint_path_is_path(self, settings: Settings) -> None:
        assert isinstance(settings.best_checkpoint_path, Path)
        assert settings.best_checkpoint_path.name == settings.BEST_CHECKPOINT_NAME

    def test_log_file_path_is_path(self, settings: Settings) -> None:
        assert isinstance(settings.log_file_path, Path)


class TestSettingsCaching:
    """Verify get_settings returns a cached singleton."""

    def test_get_settings_returns_same_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2, "get_settings() must return a cached singleton"


class TestEnsureDirectories:
    """Verify ensure_directories creates required directories."""

    def test_ensure_directories_creates_dirs(self, settings: Settings, tmp_path) -> None:
        # Override dirs to a tmp location
        s = Settings(
            CHECKPOINT_DIR=tmp_path / "ckpt",
            OUTPUT_DIR=tmp_path / "out",
            LOG_DIR=tmp_path / "logs",
            DB_PATH=tmp_path / "data" / "test.db",
        )
        s.ensure_directories()
        assert (tmp_path / "ckpt").exists()
        assert (tmp_path / "out").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "data").exists()
