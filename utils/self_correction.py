"""
Self-Correcting Training Loop for Multiphysics PINN Optimizer.

Monitors physics residuals each epoch and adaptively adjusts training
to correct constraint violations, detect divergence, and restore from
the best known checkpoint.
"""

import copy
import math
import os
import time
import torch


class PhysicsResidualTracker:
    """Tracks per-constraint physics residuals across training epochs."""

    CONSTRAINT_NAMES = ["energy_conservation", "stress_equilibrium", "em_smoothness"]

    def __init__(self, window: int = 10):
        """
        Args:
            window: Number of recent epochs used to compute moving averages.
        """
        self.window = window
        self.history: list[dict] = []  # one entry per epoch

    def record(self, epoch: int, residuals: dict[str, float], total_loss: float, data_loss: float):
        self.history.append({
            "epoch": epoch,
            "total_loss": total_loss,
            "data_loss": data_loss,
            **residuals,
        })

    def moving_avg(self, key: str, n: int | None = None) -> float | None:
        n = n or self.window
        values = [e[key] for e in self.history[-n:] if key in e]
        return sum(values) / len(values) if values else None

    def is_diverging(self, patience: int = 5, threshold: float = 1.1) -> bool:
        """Return True if total_loss has grown by >threshold× over the last `patience` epochs."""
        if len(self.history) < patience + 1:
            return False
        recent = [e["total_loss"] for e in self.history[-(patience + 1):]]
        return recent[-1] > threshold * recent[0]

    def worst_constraint(self) -> str | None:
        """Return the constraint name with the highest moving-average residual."""
        if not self.history:
            return None
        avgs = {c: self.moving_avg(c) for c in self.CONSTRAINT_NAMES if self.moving_avg(c) is not None}
        return max(avgs, key=avgs.get) if avgs else None

    def summary(self) -> dict:
        if not self.history:
            return {}
        return {k: self.moving_avg(k) for k in ["total_loss", "data_loss"] + self.CONSTRAINT_NAMES}


class SelfCorrectionLoop:
    """
    Wraps a PINNTrainer and adds a self-correcting outer loop.

    Correction strategies applied automatically:
    - **Weight boost**: increase `physics_weight` when a constraint residual
      exceeds its target tolerance.
    - **LR cooldown**: halve the learning rate when divergence is detected and
      restore from the best checkpoint.
    - **Checkpoint restore**: roll back to the best saved weights whenever the
      loss spikes above a safety threshold.

    Usage::

        from utils.self_correction import SelfCorrectionLoop

        loop = SelfCorrectionLoop(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        history = loop.run(epochs=100)
    """

    def __init__(
        self,
        trainer,
        train_loader,
        val_loader,
        *,
        physics_weight_init: float = 0.1,
        physics_weight_max: float = 5.0,
        weight_boost_factor: float = 1.5,
        residual_tolerance: float = 0.05,
        divergence_patience: int = 5,
        divergence_threshold: float = 1.15,
        lr_cooldown_factor: float = 0.5,
        restore_on_spike: bool = True,
        spike_threshold: float = 2.0,
        checkpoint_dir: str = "checkpoints",
        verbose: bool = True,
    ):
        """
        Args:
            trainer: A PINNTrainer instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            physics_weight_init: Starting weight for physics loss term.
            physics_weight_max: Maximum allowed physics weight.
            weight_boost_factor: Multiply physics_weight by this when a residual
                exceeds tolerance.
            residual_tolerance: Acceptable moving-average physics residual.
            divergence_patience: Epochs to look back when checking for divergence.
            divergence_threshold: Loss growth ratio considered divergent.
            lr_cooldown_factor: Factor to reduce LR by when divergence detected.
            restore_on_spike: Whether to restore best checkpoint on loss spike.
            spike_threshold: Loss spike multiplier vs. best loss that triggers restore.
            checkpoint_dir: Directory for saving model checkpoints.
            verbose: Print correction events.
        """
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.physics_weight = physics_weight_init
        self.physics_weight_max = physics_weight_max
        self.weight_boost_factor = weight_boost_factor
        self.residual_tolerance = residual_tolerance
        self.divergence_patience = divergence_patience
        self.divergence_threshold = divergence_threshold
        self.lr_cooldown_factor = lr_cooldown_factor
        self.restore_on_spike = restore_on_spike
        self.spike_threshold = spike_threshold
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        self.tracker = PhysicsResidualTracker()
        self.best_val_loss: float = math.inf
        self.best_weights: dict | None = None
        self.correction_log: list[dict] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, epochs: int) -> list[dict]:
        """
        Run the self-correcting training loop.

        Args:
            epochs: Total number of training epochs.

        Returns:
            List of per-epoch metric dicts.
        """
        self._log(f"Starting self-correcting loop for {epochs} epochs "
                  f"(initial physics_weight={self.physics_weight:.4f})")

        history = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.trainer.train_epoch(self.train_loader,
                                                  physics_weight=self.physics_weight)
            val_loss = self.trainer.evaluate(self.val_loader)
            residuals = self._extract_residuals()
            elapsed = time.perf_counter() - t0

            self.tracker.record(epoch, residuals, train_loss, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch)

            corrections = self._apply_corrections(epoch, val_loss, residuals)

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "physics_weight": self.physics_weight,
                "elapsed_s": elapsed,
                "corrections": corrections,
                **residuals,
            }
            history.append(entry)

            if self.verbose:
                self._print_epoch(entry)

        self._log("Training complete. "
                  f"Best val loss: {self.best_val_loss:.6f}")
        return history

    def load_best(self):
        """Load the best saved weights back into the trainer's model."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.trainer.device)
            self.trainer.model.load_state_dict(state)
            self._log("Loaded best checkpoint from disk.")
        elif self.best_weights is not None:
            self.trainer.model.load_state_dict(self.best_weights)
            self._log("Loaded best checkpoint from memory.")
        else:
            self._log("No checkpoint available to restore.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_corrections(self, epoch: int, val_loss: float, residuals: dict) -> list[str]:
        corrections = []

        # 1. Physics weight boost when a constraint is consistently violated
        worst = self.tracker.worst_constraint()
        if worst and self.tracker.moving_avg(worst, n=5) is not None:
            avg_residual = self.tracker.moving_avg(worst, n=5)
            if avg_residual > self.residual_tolerance and self.physics_weight < self.physics_weight_max:
                new_weight = min(self.physics_weight * self.weight_boost_factor,
                                 self.physics_weight_max)
                msg = (f"Epoch {epoch}: boosting physics_weight "
                       f"{self.physics_weight:.4f} → {new_weight:.4f} "
                       f"('{worst}' avg residual={avg_residual:.4f})")
                self.physics_weight = new_weight
                corrections.append(msg)
                self._record_correction(epoch, "weight_boost", msg)

        # 2. Divergence detection → LR cooldown + checkpoint restore
        if self.tracker.is_diverging(self.divergence_patience, self.divergence_threshold):
            self._reduce_lr(epoch, corrections)
            if self.restore_on_spike:
                self.load_best()
                corrections.append(f"Epoch {epoch}: restored best checkpoint after divergence")

        # 3. Spike guard: single-epoch loss spike above best
        elif (self.restore_on_spike and
              self.best_val_loss < math.inf and
              val_loss > self.spike_threshold * self.best_val_loss):
            msg = (f"Epoch {epoch}: loss spike detected "
                   f"(val={val_loss:.4f} > {self.spike_threshold}× best={self.best_val_loss:.4f})")
            corrections.append(msg)
            self._record_correction(epoch, "spike_restore", msg)
            self.load_best()

        return corrections

    def _reduce_lr(self, epoch: int, corrections: list[str]):
        for group in self.trainer.optimizer.param_groups:
            old_lr = group["lr"]
            group["lr"] = old_lr * self.lr_cooldown_factor
            msg = (f"Epoch {epoch}: divergence detected, LR "
                   f"{old_lr:.2e} → {group['lr']:.2e}")
            corrections.append(msg)
            self._record_correction(epoch, "lr_cooldown", msg)

    def _extract_residuals(self) -> dict[str, float]:
        """
        Run one no-grad forward pass over the val loader to measure
        per-constraint physics residuals.
        """
        model = self.trainer.model
        device = self.trainer.device
        model.eval()

        sums = {c: 0.0 for c in PhysicsResidualTracker.CONSTRAINT_NAMES}
        n_batches = 0

        with torch.no_grad():
            for x, y_thermal, y_stress, y_em in self.val_loader:
                x = x.to(device)
                thermal_pred, stress_pred, em_pred = model(x)

                # Mirror the three constraints from PINNModel.physics_loss
                sums["energy_conservation"] += torch.mean(
                    torch.abs(thermal_pred + em_pred * 0.1)
                ).item()
                sums["stress_equilibrium"] += torch.mean(
                    torch.relu(-stress_pred)
                ).item()
                if em_pred.shape[0] > 1:
                    sums["em_smoothness"] += torch.mean(
                        torch.abs(torch.diff(em_pred, dim=0))
                    ).item()
                n_batches += 1

        if n_batches == 0:
            return {c: 0.0 for c in sums}
        return {c: v / n_batches for c, v in sums.items()}

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save(self.trainer.model.state_dict(), path)
        self.best_weights = copy.deepcopy(self.trainer.model.state_dict())

    def _record_correction(self, epoch: int, kind: str, message: str):
        self.correction_log.append({"epoch": epoch, "kind": kind, "message": message})
        self._log(message)

    def _log(self, message: str):
        if self.verbose:
            print(f"[SelfCorrection] {message}")

    @staticmethod
    def _print_epoch(entry: dict):
        corrections = entry.get("corrections", [])
        flag = " *** CORRECTION ***" if corrections else ""
        print(
            f"Epoch {entry['epoch']:4d} | "
            f"train={entry['train_loss']:.4f} | "
            f"val={entry['val_loss']:.4f} | "
            f"pw={entry['physics_weight']:.4f} | "
            f"energy={entry.get('energy_conservation', 0):.4f} | "
            f"stress={entry.get('stress_equilibrium', 0):.4f} | "
            f"em={entry.get('em_smoothness', 0):.4f} | "
            f"{entry['elapsed_s']:.1f}s"
            f"{flag}"
        )
