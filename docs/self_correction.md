# Self-Correcting Training Loop

## Purpose

Standard PINN training uses a fixed `physics_weight` and has no mechanism to
recover from constraint violations or training instability. The
`SelfCorrectionLoop` adds three automatic correction strategies that fire
during training without any manual intervention.

## Correction Strategies

### 1. Physics Weight Boost

**Trigger**: the 5-epoch moving average of any single constraint residual
exceeds `residual_tolerance` (default 0.05).

**Action**: multiply `physics_weight` by `weight_boost_factor` (default 1.5),
capped at `physics_weight_max` (default 5.0).

**Effect**: forces the optimizer to prioritise physical consistency when the
model is violating constraints persistently.

### 2. Divergence Detection + LR Cooldown

**Trigger**: total loss at the end of the window is more than
`divergence_threshold` (1.15×) the loss at the start of the
`divergence_patience` (5-epoch) window.

**Action**:
1. Halve the learning rate for all parameter groups (`lr_cooldown_factor=0.5`).
2. Restore model weights from the best checkpoint saved so far.

**Effect**: breaks out of a loss-increasing spiral without stopping training.

### 3. Spike Guard

**Trigger**: single-epoch validation loss exceeds `spike_threshold` (2.0×)
the best validation loss ever seen.

**Action**: restore model weights from best checkpoint.

**Effect**: prevents a single bad batch from permanently derailing the model.

---

## Class: `SelfCorrectionLoop`

```python
from utils.self_correction import SelfCorrectionLoop

loop = SelfCorrectionLoop(
    trainer=trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    # --- key tuning parameters ---
    physics_weight_init=0.1,
    physics_weight_max=5.0,
    weight_boost_factor=1.5,
    residual_tolerance=0.05,
    divergence_patience=5,
    divergence_threshold=1.15,
    lr_cooldown_factor=0.5,
    restore_on_spike=True,
    spike_threshold=2.0,
    checkpoint_dir="checkpoints",
    verbose=True,
)
history = loop.run(epochs=100)
loop.load_best()   # load best weights into trainer.model
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `physics_weight_init` | 0.1 | Starting physics loss weight |
| `physics_weight_max` | 5.0 | Maximum physics loss weight |
| `weight_boost_factor` | 1.5 | Multiplier applied per boost event |
| `residual_tolerance` | 0.05 | Per-constraint acceptable residual |
| `divergence_patience` | 5 | Epoch window for divergence check |
| `divergence_threshold` | 1.15 | Loss growth ratio triggering cooldown |
| `lr_cooldown_factor` | 0.5 | LR multiplier on divergence |
| `restore_on_spike` | True | Restore checkpoint on spike/divergence |
| `spike_threshold` | 2.0 | Single-epoch spike multiplier vs best |
| `checkpoint_dir` | "checkpoints" | Directory for best model checkpoint |

---

## History & Correction Log

`loop.run()` returns a list of dicts, one per epoch:

```python
{
    "epoch": int,
    "train_loss": float,
    "val_loss": float,
    "physics_weight": float,
    "elapsed_s": float,
    "corrections": [str, ...],       # human-readable correction messages
    "energy_conservation": float,
    "stress_equilibrium": float,
    "em_smoothness": float,
}
```

All correction events are also available in `loop.correction_log`:

```python
[{"epoch": int, "kind": str, "message": str}, ...]
```

`kind` values: `"weight_boost"`, `"lr_cooldown"`, `"spike_restore"`.

---

## Tuning Guide

| Symptom | Recommended adjustment |
|---------|----------------------|
| Physics weight keeps boosting to max | Lower `residual_tolerance` or increase data quality |
| Too many LR cooldowns, slow convergence | Increase `divergence_threshold` (e.g. 1.3) |
| Restores happen too often | Increase `spike_threshold` (e.g. 3.0) |
| Constraint violations persist at end | Increase `physics_weight_max` |
| Energy conservation residual dominates | Adjust EM coupling coefficient (0.1 in C1) |
