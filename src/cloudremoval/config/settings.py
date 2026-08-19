"""
Centralised application settings using Pydantic BaseSettings.

All machine-specific values come from environment variables or a .env file.
No hardcoded absolute paths exist anywhere else in the codebase.

Usage
-----
    from cloudremoval.config import get_settings

    settings = get_settings()
    print(settings.DATASET_ROOT)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide configuration.

    Values are loaded (in priority order) from:
    1. Real environment variables
    2. A `.env` file at the project root
    3. Defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    DATASET_ROOT: Path = Field(
        default=Path("D:/allclear_test_proi1_v1"),
        description="Absolute path to the ALLClear dataset root directory.",
    )
    METADATA_FILE: str = Field(
        default="allclear_test_metadata.json",
        description="Filename of the ALLClear metadata JSON (relative to DATASET_ROOT).",
    )
    ALLCLEAR_SUBDIR: str = Field(
        default="allclear_dataset",
        description="Sub-directory inside DATASET_ROOT that holds the roi* folders.",
    )

    # ------------------------------------------------------------------
    # Project directories (relative to project root by default)
    # ------------------------------------------------------------------
    CHECKPOINT_DIR: Path = Field(
        default=Path("checkpoints"),
        description="Directory where model checkpoints are saved.",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("outputs"),
        description="Directory for inference outputs (GeoTIFFs, PNGs, etc.).",
    )
    LOG_DIR: Path = Field(
        default=Path("logs"),
        description="Directory for application log files.",
    )
    DB_PATH: Path = Field(
        default=Path("data/cloudremoval.db"),
        description="SQLite database file path.",
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    DEVICE: str = Field(
        default="cuda",
        description="PyTorch device string: 'cuda' or 'cpu'.",
    )

    # ------------------------------------------------------------------
    # Training (Phase 2+ placeholders - MAX_EPOCHS is locked at 30)
    # ------------------------------------------------------------------
    MAX_EPOCHS: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Maximum training epochs. Default 30; can be extended up to 200 via CLI.",
    )
    BATCH_SIZE: int = Field(
        default=4,
        ge=1,
        description="Training batch size.",
    )
    LEARNING_RATE: float = Field(
        default=1e-4,
        gt=0,
        description="Initial learning rate.",
    )
    WEIGHT_DECAY: float = Field(
        default=1e-5,
        ge=0,
        description="Optimizer weight decay.",
    )
    VAL_FREQ: int = Field(
        default=1,
        ge=1,
        description="Validate every N epochs.",
    )
    AMP: bool = Field(
        default=True,
        description="Use automatic mixed precision (AMP) during training.",
    )
    NUM_WORKERS: int = Field(
        default=4,
        ge=0,
        description="DataLoader worker processes.",
    )
    MISSING_S1_STRATEGY: Literal["skip", "zeros"] = Field(
        default="skip",
        description=(
            "Strategy when an S1 image is absent for a sample: "
            "'skip' ignores the sample; 'zeros' substitutes a zero tensor."
        ),
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    BEST_CHECKPOINT_NAME: str = Field(
        default="best_model.pth",
        description="Filename of the best checkpoint inside CHECKPOINT_DIR.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Root logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    LOG_FILE: str = Field(
        default="cloudremoval.log",
        description="Log filename (relative to LOG_DIR).",
    )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host.")
    API_PORT: int = Field(default=8000, description="FastAPI port.")
    API_DEBUG: bool = Field(default=False, description="Enable FastAPI debug mode.")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cuda", "cpu"}
        if v not in allowed:
            raise ValueError(f"DEVICE must be one of {allowed}, got '{v}'")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return v_upper

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def metadata_path(self) -> Path:
        """Absolute path to the ALLClear metadata JSON."""
        return self.DATASET_ROOT / self.METADATA_FILE

    @property
    def allclear_dataset_path(self) -> Path:
        """Absolute path to the allclear_dataset sub-directory."""
        return self.DATASET_ROOT / self.ALLCLEAR_SUBDIR

    @property
    def best_checkpoint_path(self) -> Path:
        """Absolute path to the best model checkpoint."""
        return self.CHECKPOINT_DIR / self.BEST_CHECKPOINT_NAME

    @property
    def log_file_path(self) -> Path:
        """Absolute path to the log file."""
        return self.LOG_DIR / self.LOG_FILE

    def ensure_directories(self) -> None:
        """Create project output directories if they do not already exist."""
        for directory in (self.CHECKPOINT_DIR, self.OUTPUT_DIR, self.LOG_DIR, self.DB_PATH.parent):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env on first call)."""
    return Settings()
