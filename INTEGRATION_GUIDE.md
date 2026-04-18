# DATA MANAGER INTEGRATION GUIDE
## Changes to master_multi_physics_pipeline.py

---

## SUMMARY OF CHANGES:

✅ Added PINNOptimizationDataManager import
✅ Integrated data manager into run_pipeline()
✅ Connected data manager to training loop
✅ Added story logging for all major events
✅ Automatic HDF5 + JSON storage for all runs

---

## STEP-BY-STEP CHANGES:

### 1. ADD IMPORT (Line 16)

**OLD:**
```python
from models.pinn_model import PINNModel, PINNTrainer
from utils.self_correction import SelfCorrectionLoop
```

**NEW:**
```python
from models.pinn_model import PINNModel, PINNTrainer
from utils.self_correction import SelfCorrectionLoop
from utils.pinn_data_manager import PINNOptimizationDataManager  # NEW!
```

---

### 2. ADD data_manager PARAMETER TO run_training() (Line 54)

**OLD:**
```python
def run_training(
    train_loader,
    val_loader,
    *,
    input_dim: int = 10,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    physics_weight_init: float = 0.1,
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
):
```

**NEW:**
```python
def run_training(
    train_loader,
    val_loader,
    *,
    input_dim: int = 10,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    physics_weight_init: float = 0.1,
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
    data_manager=None,  # NEW! Pass in data manager instance
):
```

---

### 3. REPLACE loop.run() WITH MANUAL LOOP (Lines 90-150)

**OLD:**
```python
history = loop.run(epochs=epochs)

# Restore best weights before returning
loop.load_best()

return model, history
```

**NEW:**
```python
# =================================================================
# TRAINING LOOP WITH DATA MANAGER INTEGRATION
# =================================================================

for epoch in range(epochs):
    # Train one epoch
    train_metrics = loop.train_epoch(epoch)
    val_loss = loop.validate()
    
    # Check for improvement
    if val_loss < loop.best_val_loss:
        loop.best_val_loss = val_loss
        loop.best_epoch = epoch
        loop.patience_counter = 0
        loop.save_checkpoint(epoch, is_best=True)
        
        # Save to data manager if provided
        if data_manager is not None:
            data_manager.save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                is_best=True
            )
    else:
        loop.patience_counter += 1
    
    # Self-correction logic
    if loop.should_correct():
        loop.adjust_physics_weight(
            epoch,
            val_loss,
            train_metrics['physics_loss'],
            train_metrics['data_loss']
        )
        
        # Log correction to data manager
        if data_manager is not None and len(loop.correction_log) > 0:
            correction = loop.correction_log[-1]
            data_manager.log_story_event(
                event_type="correction",
                description=f"Adjusted physics weight: {correction['old_weight']:.4f} → {correction['new_weight']:.4f}",
                epoch=epoch,
                data=correction
            )
    
    # Log training epoch to data manager
    if data_manager is not None:
        data_manager.append_training_epoch(
            epoch=epoch,
            train_loss=train_metrics['train_loss'],
            val_loss=val_loss,
            physics_loss=train_metrics['physics_loss'],
            data_loss=train_metrics['data_loss'],
            learning_rate=trainer.optimizer.param_groups[0]['lr'],
            physics_weight=loop.physics_weight
        )
    
    # Periodic checkpoint
    if epoch % 50 == 0:
        loop.save_checkpoint(epoch, is_best=False)

# Restore best weights before returning
loop.load_best()

# Get history
history = loop.history

return model, history
```

---

### 4. COMPLETE RUN_PIPELINE() REWRITE (Lines 152-230)

**ADDED:**
- Data manager initialization
- Run ID logging
- Simulation event logging
- Training completion logging
- Final model saving

See the full updated file for complete implementation.

---

## HOW TO APPLY THESE CHANGES:

### Option 1: Replace Your Entire File
1. Download: `master_multi_physics_pipeline_UPDATED.py`
2. Rename to: `master_multi_physics_pipeline.py`
3. Replace your current file

### Option 2: Manual Edits
1. Follow each step above
2. Copy/paste the NEW code
3. Save the file

---

## WHAT YOU GET:

After integration, every `run_pipeline()` call will:

✅ Create timestamped run directory
✅ Save all simulation results to HDF5 (when you add real data)
✅ Log every training epoch with metrics
✅ Log every self-correction decision to story.json
✅ Save model checkpoints
✅ Generate complete training history
✅ Compress data for storage efficiency

**Output structure:**
```
pinn_optimization_runs/
└── run_20260405_183045/
    ├── simulation_results.h5    (CFD/Thermal/EM data)
    ├── training_history.h5      (Loss curves, metrics)
    ├── story.json                (Self-correction narrative)
    ├── checkpoints/
    │   ├── best_model.pt
    │   └── epoch_050.pt
    └── metadata.json
```

---

## TESTING:

```python
# Run the updated pipeline
from utils.data_loader import create_dataloaders

train_loader, val_loader = create_dataloaders(batch_size=32)
model, history = run_pipeline(train_loader, val_loader, epochs=100)

# Check the output
print(f"Run saved to: pinn_optimization_runs/run_*/")
```

---

## QUESTIONS?

- Check `docs/PINN_DATA_STORAGE_GUIDE-1.md` for data manager API
- Check `examples/complete_pipeline_example.py` for full workflow
- Self-correction loop docs: `docs/self_correction.md`
