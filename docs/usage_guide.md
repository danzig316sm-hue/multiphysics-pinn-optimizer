# Usage Guide

## 1. Prepare Your Data

The data loader expects a CSV with:
- All parameter columns first (any number)
- Last 3 columns: `thermal`, `stress`, `em` (in that order)

Example CSV layout:

```
length,width,height,...,thermal,stress,em
1.2,0.8,0.15,...,312.4,5.2e6,0.034
...
```

## 2. Load Data

```python
from utils.data_loader import create_dataloader

train_loader, val_loader, dataset = create_dataloader(
    data_file="data/my_data.csv",
    batch_size=32,
    test_split=0.2,
)

# Save scalers for later inference
dataset.save_scalers("checkpoints/scalers")
```

## 3. Train with Self-Correction

### Option A — Full pipeline (recommended)

```python
from master_multi_physics_pipeline import run_pipeline

model, history = run_pipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
)
```

### Option B — Manual control

```python
from models.pinn_model import PINNModel, PINNTrainer
from utils.self_correction import SelfCorrectionLoop

model = PINNModel(input_dim=10)
trainer = PINNTrainer(model, learning_rate=1e-3)

loop = SelfCorrectionLoop(
    trainer=trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    physics_weight_init=0.1,
    residual_tolerance=0.02,   # tighter tolerance
)
history = loop.run(epochs=200)
loop.load_best()
```

## 4. Inspect Training History

```python
import json

# Print correction events
for entry in history:
    if entry["corrections"]:
        print(f"Epoch {entry['epoch']}: {entry['corrections']}")

# Plot loss curves (requires matplotlib)
import matplotlib.pyplot as plt
epochs = [e["epoch"] for e in history]
plt.plot(epochs, [e["train_loss"] for e in history], label="train")
plt.plot(epochs, [e["val_loss"] for e in history], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

## 5. Run Inference

```python
import torch
from models.pinn_model import PINNModel
from utils.data_loader import MultiPhysicsDataset

# Load model
model = PINNModel(input_dim=10)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# Load scalers
scalers = MultiPhysicsDataset.load_scalers("checkpoints/scalers")

# Prepare input
raw_input = [[1.2, 0.8, 0.15, ...]]   # shape (1, input_dim)
x = torch.tensor(scalers["input"].transform(raw_input), dtype=torch.float32)

with torch.no_grad():
    thermal, stress, em = model(x)

# Inverse-transform predictions
thermal_value = scalers["thermal"].inverse_transform(thermal.numpy())
stress_value  = scalers["stress"].inverse_transform(stress.numpy())
em_value      = scalers["em"].inverse_transform(em.numpy())
```

## 6. Save & Restore

Best model checkpoint is saved automatically to `checkpoints/best_model.pt`
during `SelfCorrectionLoop.run()`. To manually save:

```python
torch.save(model.state_dict(), "my_checkpoint.pt")
```

## 7. Google Drive (Colab)

```python
from utils.data_loader import download_from_google_drive, upload_to_google_drive

# Download dataset
download_from_google_drive(file_id="YOUR_FILE_ID", destination="data/my_data.csv")

# Upload results (Colab only)
upload_to_google_drive("checkpoints/best_model.pt", folder_id="YOUR_FOLDER_ID")
```

## 8. Browse Engineering Docs

```bash
python docs_viewer.py list
python docs_viewer.py show physics
python docs_viewer.py search "energy conservation"
```
