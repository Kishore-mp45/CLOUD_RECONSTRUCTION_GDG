"""
src/cloudremoval/training/train_logger.py
==========================================
Persistent training logger for Phase 4.

Creates a timestamped log directory:

    logs/training/YYYYMMDD_HHMMSS/
    ├── training.log           <- plaintext human-readable log
    ├── training_config.json   <- full config snapshot at run start
    ├── training_metrics.csv   <- one row per epoch (appended each epoch)
    └── training_metrics.json  <- complete metrics list (written at end)

Usage:
    logger = TrainingLogger(log_root=Path("logs/training"))
    logger.start(config_dict)           # called once before training
    logger.log_epoch(epoch_record)      # called after every epoch
    logger.finish(summary_dict)         # called once after training
    logger.close()                      # cleanup file handles

All methods are safe to call even if the log dir creation fails
(degrades gracefully to stderr logging only).
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# CSV column order (matches Phase 4 spec)
_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "psnr",
    "ssim",
    "mae",
    "rmse",
    "learning_rate",
    "epoch_time_s",
    "elapsed_time_s",
    "gpu_memory_reserved_gb",
    "is_best",
    "early_stopping_counter",
]


class TrainingLogger:
    """Manages log directory, CSV metrics, JSON config, and file handlers.

    Parameters
    ----------
    log_root : Path
        Root directory for training logs (e.g. Path("logs/training")).
        A timestamped subdirectory is created inside this root.

    Attributes
    ----------
    run_dir : Path
        The unique timestamped directory for this training run.
    """

    def __init__(self, log_root: Path) -> None:
        self._log_root = log_root
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = log_root / self._timestamp
        self._file_handler: Optional[logging.FileHandler] = None
        self._csv_file = None
        self._csv_writer = None
        self._all_epochs: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, config_dict: dict) -> None:
        """Create log directory and write config snapshot.

        Must be called once before the training loop begins.

        Parameters
        ----------
        config_dict : dict
            Full training + model configuration to snapshot.
        """
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)

            # --- File log handler ---
            log_file = self.run_dir / "training.log"
            self._file_handler = logging.FileHandler(log_file, encoding="utf-8")
            self._file_handler.setLevel(logging.DEBUG)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            self._file_handler.setFormatter(fmt)
            logging.getLogger().addHandler(self._file_handler)

            # --- Config snapshot ---
            config_path = self.run_dir / "training_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, default=str)

            # --- CSV header ---
            csv_path = self.run_dir / "training_metrics.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=_CSV_FIELDS,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()

            log.info("Training log directory: %s", self.run_dir)
            print(f"  Log directory : {self.run_dir}")

        except Exception as exc:
            log.warning("Could not create log directory: %s", exc)

    def log_epoch(self, record: dict) -> None:
        """Append one epoch's metrics to the CSV and in-memory list.

        Parameters
        ----------
        record : dict
            Keys should match _CSV_FIELDS. Extra keys are ignored.
        """
        self._all_epochs.append(record)
        try:
            if self._csv_writer is not None:
                self._csv_writer.writerow(record)
                self._csv_file.flush()
        except Exception as exc:
            log.warning("Failed to write epoch metrics to CSV: %s", exc)

    def finish(self, summary: dict) -> None:
        """Write final JSON metrics file and summary.

        Parameters
        ----------
        summary : dict
            Overall training summary (best epoch, best metrics, etc.)
        """
        try:
            # Complete JSON metrics
            json_path = self.run_dir / "training_metrics.json"
            payload = {
                "summary": summary,
                "epochs": self._all_epochs,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            log.info("Training metrics JSON: %s", json_path)

        except Exception as exc:
            log.warning("Failed to write training_metrics.json: %s", exc)

    def close(self) -> None:
        """Close file handles."""
        try:
            if self._csv_file is not None:
                self._csv_file.close()
            if self._file_handler is not None:
                logging.getLogger().removeHandler(self._file_handler)
                self._file_handler.close()
        except Exception as exc:
            log.warning("Error closing log handles: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def log_file(self) -> Path:
        return self.run_dir / "training.log"

    @property
    def csv_file(self) -> Path:
        return self.run_dir / "training_metrics.csv"

    @property
    def config_file(self) -> Path:
        return self.run_dir / "training_config.json"

    @property
    def json_file(self) -> Path:
        return self.run_dir / "training_metrics.json"
