"""
src/cloudremoval/training/trainer.py
=======================================
Phase 4 Training loop for DSen2-CR cloud removal.

Features:
  - AMP (Automatic Mixed Precision) — torch.amp, PyTorch 2.5 API
  - Gradient accumulation
  - Phase 4 verbose terminal output (matches spec exactly)
  - Train + validation loss with full metrics every epoch
  - EarlyStopping integration (patience=5 by default)
  - TrainingLogger integration (CSV + JSON + file log)
  - Checkpoint: latest.pth + best_model.pth + epoch_NNN.pth every epoch
  - Resume from checkpoint (restores model, optimizer, scheduler, early stopping)
  - GPU memory reporting every epoch
  - Elapsed time tracking
  - Best epoch tracking
  - NaN/Inf guard — raises explicitly on corrupted loss
  - CUDA OOM guard — reports and re-raises (never silently switches to CPU)
  - Post-training verification block

Usage (via scripts/train.py — do NOT call this directly):
    trainer = Trainer(
        model, criterion, train_loader, val_loader,
        optimizer, config, checkpoint_dir,
        early_stopping=es, logger=tl, resume_path=path,
    )
    result = trainer.train()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cloudremoval.models.model_config import DSen2CRConfig
from cloudremoval.models.losses import CloudRemovalLoss
from cloudremoval.models.checkpoint import save_checkpoint, load_checkpoint
from cloudremoval.training.metrics import compute_all_metrics, MetricTracker
from cloudremoval.training.early_stopping import EarlyStopping
from cloudremoval.training.train_logger import TrainingLogger

log = logging.getLogger(__name__)


class NaNLossError(RuntimeError):
    """Raised when loss becomes NaN or Inf during training."""


class Trainer:
    """Phase 4 full training loop for DSen2-CR.

    Parameters
    ----------
    model : nn.Module
        DSen2CR model (already moved to device).
    criterion : CloudRemovalLoss
        Loss function (already moved to device).
    train_loader : DataLoader
    val_loader : DataLoader
    optimizer : torch.optim.Optimizer
    config : DSen2CRConfig
    checkpoint_dir : Path
    early_stopping : EarlyStopping | None
        If None, training runs for all max_epochs without early stopping.
    logger : TrainingLogger | None
        If None, only stdout output is produced.
    resume_path : Path | None
        If provided, load this checkpoint and continue training.
    max_epochs : int | None
        Override config.max_epochs (used for CLI --epochs).
    scheduler : torch.optim.lr_scheduler._LRScheduler | None
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: CloudRemovalLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: "torch.optim.Optimizer",
        config: DSen2CRConfig,
        checkpoint_dir: Path,
        early_stopping: Optional[EarlyStopping] = None,
        logger: Optional[TrainingLogger] = None,
        resume_path: Optional[Path] = None,
        max_epochs: Optional[int] = None,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
    ) -> None:
        self.model          = model
        self.criterion      = criterion
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.config         = config
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping = early_stopping
        self.logger         = logger
        self.device         = torch.device(config.device)
        self.amp_enabled    = config.amp_enabled and config.device == "cuda"
        self.grad_accum     = max(1, config.grad_accum_steps)
        self.max_epochs     = max_epochs if max_epochs is not None else config.max_epochs

        # AMP scaler
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.amp_enabled:
            self.scaler = torch.amp.GradScaler(device="cuda")

        # Training state (may be overwritten by resume)
        self.start_epoch:   int   = 1
        self.best_val_loss: float = float("inf")
        self.best_epoch:    int   = 0
        self.best_metrics:  dict  = {}

        # Resume
        if resume_path is not None:
            self._resume(resume_path)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resume(self, path: Path) -> None:
        print(f"\n{'='*60}")
        print(f"  RESUMING FROM CHECKPOINT")
        print(f"  Path  : {path}")
        print(f"{'='*60}")

        ckpt = load_checkpoint(
            checkpoint_path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )
        self.start_epoch   = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        self.best_epoch    = ckpt.get("epoch",    0)

        # Restore early stopping counter
        if self.early_stopping is not None and "early_stopping" in ckpt:
            es_state = ckpt.get("early_stopping")
            if es_state:
                self.early_stopping.load_state_dict(es_state)

        print(f"  Resumed from epoch : {ckpt['epoch']}")
        print(f"  Continuing from    : epoch {self.start_epoch}")
        print(f"  Best val loss      : {self.best_val_loss:.6f}")
        if self.early_stopping is not None:
            print(f"  Early stopping     : {self.early_stopping}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full Phase 4 training loop.

        Returns
        -------
        dict
            Summary with best_val_loss, best_epoch, best_metrics, etc.
        """
        n_train = len(self.train_loader.dataset)
        n_val   = len(self.val_loader.dataset)

        if torch.cuda.is_available():
            gpu_name   = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        else:
            gpu_name   = "CPU"
            vram_total = 0.0

        es_patience = self.early_stopping.patience if self.early_stopping else "disabled"

        print(f"\n{'='*60}")
        print("  DSEN2-CR TRAINING")
        print(f"{'='*60}")
        print(f"  GPU              : {gpu_name}")
        print(f"  VRAM             : {vram_total:.1f} GB")
        print(f"  AMP              : {'ENABLED' if self.amp_enabled else 'DISABLED'}")
        print(f"  Epochs           : {self.start_epoch} \u2192 {self.max_epochs}")
        print(f"  Train samples    : {n_train:,}")
        print(f"  Val samples      : {n_val:,}")
        print(f"  Batch size       : {self.config.batch_size}")
        print(f"  Grad accum       : {self.grad_accum}  (eff. batch={self.config.batch_size * self.grad_accum})")
        print(f"  Learning rate    : {self.config.learning_rate}")
        print(f"  Weight decay     : {self.config.weight_decay}")
        print(f"  Early stopping   : patience={es_patience}")
        print(f"  Checkpoint dir   : {self.checkpoint_dir}")
        if self.logger is not None:
            print(f"  Log dir          : {self.logger.run_dir}")
        print(f"{'='*60}")

        total_t0 = time.time()
        early_stopped = False

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            epoch_t0 = time.time()
            elapsed  = epoch_t0 - total_t0

            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch}/{self.max_epochs}  |  elapsed={_fmt_time(elapsed)}")
            print(f"{'─'*60}")

            # --- Train ---
            train_metrics = self._train_epoch(epoch)

            # --- Validate ---
            val_metrics = self._val_epoch(epoch)

            # --- Scheduler ---
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Best model tracking ---
            val_loss = val_metrics.get("loss", float("inf"))
            is_best  = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                self.best_metrics  = dict(val_metrics)

            # --- Checkpoint ---
            es_state = self.early_stopping.state_dict() if self.early_stopping else None
            ckpt_paths = save_checkpoint(
                checkpoint_dir=self.checkpoint_dir,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_loss,
                train_loss=train_metrics.get("loss", 0.0),
                config_dict=self.config.to_dict(),
                norm_version=self.config.norm_version,
                amp_enabled=self.amp_enabled,
                scheduler=self.scheduler,
                early_stopping_state=es_state,
                is_best=is_best,
            )
            ckpt_tag = "latest"
            if is_best:
                ckpt_tag += " + BEST_MODEL"

            # --- GPU memory ---
            epoch_time = time.time() - epoch_t0
            elapsed    = time.time() - total_t0
            mem_str    = self._gpu_memory_str()
            lr_now     = self.optimizer.param_groups[0]["lr"]

            # --- Log record ---
            gpu_reserved_gb = 0.0
            if torch.cuda.is_available():
                gpu_reserved_gb = torch.cuda.memory_reserved(0) / 1024 ** 3

            epoch_record = {
                "epoch":                    epoch,
                "train_loss":               round(train_metrics.get("loss", 0.0), 8),
                "val_loss":                 round(val_loss, 8),
                "psnr":                     round(val_metrics.get("psnr", 0.0), 4),
                "ssim":                     round(val_metrics.get("ssim", 0.0), 6),
                "mae":                      round(val_metrics.get("mae",  0.0), 8),
                "rmse":                     round(val_metrics.get("rmse", 0.0), 8),
                "learning_rate":            lr_now,
                "epoch_time_s":             round(epoch_time, 2),
                "elapsed_time_s":           round(elapsed, 2),
                "gpu_memory_reserved_gb":   round(gpu_reserved_gb, 3),
                "is_best":                  is_best,
                "early_stopping_counter":   self.early_stopping.counter if self.early_stopping else 0,
            }
            if self.logger is not None:
                self.logger.log_epoch(epoch_record)

            # --- Phase 4 terminal summary ---
            self._print_epoch_summary(
                epoch=epoch,
                train_m=train_metrics,
                val_m=val_metrics,
                is_best=is_best,
                epoch_time=epoch_time,
                elapsed=elapsed,
                mem=mem_str,
                lr=lr_now,
                ckpt_tag=ckpt_tag,
            )

            # --- Early stopping ---
            if self.early_stopping is not None:
                stop = self.early_stopping.step(val_loss, epoch=epoch)
                if stop:
                    print(f"\n  [EARLY STOP] No improvement for {self.early_stopping.patience} epochs.")
                    print(f"  [EARLY STOP] Best epoch: {self.best_epoch}  Best val loss: {self.best_val_loss:.6f}")
                    early_stopped = True
                    break

        # ------------------------------------------------------------------
        # End of training
        # ------------------------------------------------------------------
        total_time  = time.time() - total_t0
        actual_epochs = epoch - self.start_epoch + 1

        summary = {
            "epochs_requested":     self.max_epochs,
            "epochs_completed":     actual_epochs,
            "early_stopped":        early_stopped,
            "best_epoch":           self.best_epoch,
            "best_val_loss":        self.best_val_loss,
            "best_psnr":            self.best_metrics.get("psnr",  0.0),
            "best_ssim":            self.best_metrics.get("ssim",  0.0),
            "best_mae":             self.best_metrics.get("mae",   0.0),
            "best_rmse":            self.best_metrics.get("rmse",  0.0),
            "total_time_s":         total_time,
            "avg_epoch_time_s":     total_time / max(actual_epochs, 1),
        }

        if self.logger is not None:
            self.logger.finish(summary)

        # Phase 4 final banner
        self._print_final_summary(summary, early_stopped)

        return summary

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        tracker   = MetricTracker()
        n_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            s2     = batch["s2"].to(self.device, non_blocking=True)
            s1     = batch["s1"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            B      = s2.size(0)

            try:
                if self.amp_enabled and self.scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        pred = self.model(s2, s1)
                        loss, components = self.criterion(pred, target)
                        loss = loss / self.grad_accum
                    self.scaler.scale(loss).backward()
                else:
                    pred = self.model(s2, s1)
                    loss, components = self.criterion(pred, target)
                    loss = loss / self.grad_accum
                    loss.backward()

            except torch.cuda.OutOfMemoryError as oom:
                msg = (
                    f"[OOM] CUDA out of memory at epoch {epoch} batch {batch_idx+1}. "
                    f"Reduce --batch-size or increase --grad-accum. Error: {oom}"
                )
                log.error(msg)
                print(f"\n  [FATAL] {msg}")
                raise

            # NaN/Inf guard
            raw_loss = components["total"]
            if not torch.isfinite(torch.tensor(raw_loss)):
                msg = (
                    f"[NaN] Loss is {raw_loss} at epoch {epoch} batch {batch_idx+1}. "
                    f"Training aborted to prevent corrupted state."
                )
                log.error(msg)
                print(f"\n  [FATAL] {msg}")
                raise NaNLossError(msg)

            # Optimizer step after accumulation
            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == n_batches:
                if self.amp_enabled and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            tracker.update({"loss": raw_loss}, n=B)

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                lr = self.optimizer.param_groups[0]["lr"]
                pct = 100.0 * (batch_idx + 1) / n_batches
                print(
                    f"    [Train] {batch_idx+1:4d}/{n_batches} ({pct:5.1f}%)  "
                    f"loss={raw_loss:.5f}  lr={lr:.2e}",
                    flush=True,
                )

        return tracker.compute()

    # ------------------------------------------------------------------
    # Validate one epoch
    # ------------------------------------------------------------------

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                s2     = batch["s2"].to(self.device, non_blocking=True)
                s1     = batch["s1"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)
                B      = s2.size(0)

                if self.amp_enabled:
                    with torch.amp.autocast(device_type="cuda"):
                        pred = self.model(s2, s1)
                        loss, components = self.criterion(pred, target)
                else:
                    pred = self.model(s2, s1)
                    loss, components = self.criterion(pred, target)

                img_metrics = compute_all_metrics(pred, target)
                combined    = {"loss": components["total"], **img_metrics}
                tracker.update(combined, n=B)

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.val_loader):
                    pct = 100.0 * (batch_idx + 1) / len(self.val_loader)
                    print(
                        f"    [Val]   {batch_idx+1:4d}/{len(self.val_loader)} ({pct:5.1f}%)  "
                        f"loss={components['total']:.5f}",
                        flush=True,
                    )

        return tracker.compute()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gpu_memory_str(self) -> str:
        if not torch.cuda.is_available():
            return "n/a (CPU)"
        alloc    = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        peak     = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        return f"alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  peak={peak:.2f}GB"

    def _print_epoch_summary(
        self,
        epoch: int,
        train_m: dict,
        val_m: dict,
        is_best: bool,
        epoch_time: float,
        elapsed: float,
        mem: str,
        lr: float,
        ckpt_tag: str,
    ) -> None:
        best_tag = "  ◄ BEST" if is_best else ""
        es_str   = ""
        if self.early_stopping is not None:
            es_str = f"  ({self.early_stopping.counter}/{self.early_stopping.patience} patience)"

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  Epoch {epoch:3d} / {self.max_epochs}  Summary{' '*22}│")
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │  Train Loss : {train_m.get('loss',0.0):>10.6f}                   │")
        print(f"  │  Val Loss   : {val_m.get('loss',0.0):>10.6f}{best_tag:<16}│")
        print(f"  │  PSNR       : {val_m.get('psnr',0.0):>10.4f} dB                │")
        print(f"  │  SSIM       : {val_m.get('ssim',0.0):>10.6f}                   │")
        print(f"  │  MAE        : {val_m.get('mae', 0.0):>10.6f}                   │")
        print(f"  │  RMSE       : {val_m.get('rmse',0.0):>10.6f}                   │")
        print(f"  │  LR         : {lr:>10.2e}                   │")
        print(f"  │  Epoch time : {epoch_time:>10.1f}s                  │")
        print(f"  │  Elapsed    : {_fmt_time(elapsed):>10}                   │")
        print(f"  │  Best epoch : {self.best_epoch:>10d}{es_str:<16}│")
        print(f"  │  GPU memory : {mem:<31}│")
        print(f"  │  Checkpoint : {ckpt_tag:<31}│")
        print(f"  └─────────────────────────────────────────────┘")

    def _print_final_summary(self, summary: dict, early_stopped: bool) -> None:
        print(f"\n{'='*60}")
        print("  PHASE 4 COMPLETE")
        print(f"{'='*60}")
        print(f"  Epochs completed   : {summary['epochs_completed']} / {summary['epochs_requested']}")
        print(f"  Early stopping     : {'YES' if early_stopped else 'NO'}")
        print(f"  Best epoch         : {summary['best_epoch']}")
        print(f"  Best val loss      : {summary['best_val_loss']:.6f}")
        print(f"  Best PSNR          : {summary['best_psnr']:.4f} dB")
        print(f"  Best SSIM          : {summary['best_ssim']:.6f}")
        print(f"  Best MAE           : {summary['best_mae']:.6f}")
        print(f"  Best RMSE          : {summary['best_rmse']:.6f}")
        print(f"  Total time         : {_fmt_time(summary['total_time_s'])}")
        print(f"  Avg epoch time     : {summary['avg_epoch_time_s']:.1f}s")
        print(f"  Best checkpoint    : {self.checkpoint_dir / 'best_model.pth'}")
        print(f"  Latest checkpoint  : {self.checkpoint_dir / 'latest.pth'}")
        if self.logger is not None:
            print(f"  Log dir            : {self.logger.run_dir}")
            print(f"  Metrics CSV        : {self.logger.csv_file}")
        print(f"{'='*60}")


def _fmt_time(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
    return f"{s//60}:{s%60:02d}"
