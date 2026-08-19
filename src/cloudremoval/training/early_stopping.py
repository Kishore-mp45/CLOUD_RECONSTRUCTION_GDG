"""
src/cloudremoval/training/early_stopping.py
=============================================
Early stopping for DSen2-CR training.

Logic:
  - Monitors validation loss each epoch.
  - If val_loss does not improve by at least `min_delta` for `patience`
    consecutive epochs, training is stopped.
  - The best model checkpoint is preserved independently — early stopping
    only controls the training loop termination.

Phase 4 default: patience = 5

Example usage (inside Trainer):
    es = EarlyStopping(patience=5)
    for epoch in ...:
        val_loss = ...
        if es.step(val_loss):
            print("Early stopping triggered")
            break

Checkpoint integration:
    # Save state with checkpoint:
    payload["early_stopping"] = es.state_dict()

    # Restore on resume:
    es.load_state_dict(ckpt.get("early_stopping", {}))
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class EarlyStopping:
    """Monitor validation loss and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
        Default: 5.
    min_delta : float
        Minimum improvement in val_loss to count as an improvement.
        Default: 1e-6.
    mode : str
        "min" (default) → lower is better (val_loss).
        "max"           → higher is better (PSNR).

    Attributes
    ----------
    best_score : float
        Best value seen so far.
    counter : int
        Number of consecutive epochs without improvement.
    stopped : bool
        True once early stopping has been triggered.
    best_epoch : int
        Epoch number at which the best score was recorded.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-6,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.counter:    int   = 0
        self.stopped:    bool  = False
        self.best_epoch: int   = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metric: float, epoch: int = 0) -> bool:
        """Evaluate one epoch's metric.

        Parameters
        ----------
        metric : float
            Validation metric for this epoch (lower if mode='min').
        epoch : int
            Current epoch number (for tracking best_epoch).

        Returns
        -------
        bool
            True if training should stop now.
        """
        if self.stopped:
            return True

        improved = self._is_improvement(metric)

        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter    = 0
            log.debug(
                "EarlyStopping: improvement at epoch %d (best=%.6f)",
                epoch, metric,
            )
        else:
            self.counter += 1
            log.info(
                "EarlyStopping: no improvement for %d/%d epochs "
                "(current=%.6f best=%.6f)",
                self.counter, self.patience, metric, self.best_score,
            )

        if self.counter >= self.patience:
            self.stopped = True
            log.info(
                "EarlyStopping: TRIGGERED at epoch %d "
                "(no improvement for %d epochs, best epoch=%d)",
                epoch, self.patience, self.best_epoch,
            )
            return True

        return False

    def is_best(self, metric: float) -> bool:
        """Return True if `metric` is a new best (without advancing the counter)."""
        return self._is_improvement(metric)

    # ------------------------------------------------------------------
    # Serialisation (for checkpoint embedding)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serialisable state dict for checkpoint storage."""
        return {
            "patience":   self.patience,
            "min_delta":  self.min_delta,
            "mode":       self.mode,
            "best_score": self.best_score,
            "counter":    self.counter,
            "stopped":    self.stopped,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint dict."""
        if not state:
            return
        self.patience   = state.get("patience",   self.patience)
        self.min_delta  = state.get("min_delta",  self.min_delta)
        self.mode       = state.get("mode",       self.mode)
        self.best_score = state.get("best_score", self.best_score)
        self.counter    = state.get("counter",    self.counter)
        self.stopped    = state.get("stopped",    self.stopped)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        log.info(
            "EarlyStopping: restored (counter=%d/%d best_epoch=%d best=%.6f)",
            self.counter, self.patience, self.best_epoch, self.best_score,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_score - self.min_delta
        else:
            return metric > self.best_score + self.min_delta

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode={self.mode}, "
            f"counter={self.counter}/{self.patience}, "
            f"best={self.best_score:.6f} @ epoch {self.best_epoch}, "
            f"stopped={self.stopped})"
        )
