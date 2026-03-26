# Network Architecture

## Overview

The PINN uses a shared encoder trunk with three domain-specific output heads.
All three physics outputs are computed in a single forward pass.

```
Input (batch × input_dim)
        │
        ▼
┌───────────────────┐
│  Shared Encoder   │  Linear(input_dim → 128) → ReLU
│                   │  Linear(128 → 256)        → ReLU
│                   │  Linear(256 → 128)        → ReLU
└────────┬──────────┘
         │ (batch × 128)
    ┌────┼────┐
    ▼    ▼    ▼
 Thermal Stress  EM
  Head   Head  Head
  Lin(128→1) each
    │    │    │
thermal stress  em
 pred  pred   pred
```

## Layer Details

| Layer | In | Out | Activation |
|-------|----|-----|------------|
| encoder[0] | input_dim (default 10) | 128 | ReLU |
| encoder[1] | 128 | 256 | ReLU |
| encoder[2] | 256 | 128 | ReLU |
| thermal_head | 128 | 1 | none |
| stress_head | 128 | 1 | none |
| em_head | 128 | 1 | none |

Hidden dimensions are configurable via `hidden_dims` at construction time.

## Class API

### `PINNModel(input_dim, hidden_dims, output_dims)`

| Arg | Default | Description |
|-----|---------|-------------|
| `input_dim` | `10` | Number of parametric input features |
| `hidden_dims` | `[128, 256, 128]` | Width of each encoder layer |
| `output_dims` | `{'thermal':1,'stress':1,'EM':1}` | Output size per head |

**`forward(x)`** → `(thermal_pred, stress_pred, em_pred)`

**`physics_loss(..., physics_weight)`** → `(total_loss, data_loss, physics_loss, residuals_dict)`

- `residuals_dict` keys: `energy_conservation`, `stress_equilibrium`, `em_smoothness`

### `PINNTrainer(model, learning_rate, device)`

| Method | Description |
|--------|-------------|
| `train_epoch(dataloader, physics_weight)` | One full training epoch, returns avg loss |
| `evaluate(dataloader)` | No-grad eval pass, returns avg loss |

Optimizer: Adam. Scheduler: StepLR (step=10, γ=0.9).

## Data Flow

```
CSV file
  └─► MultiPhysicsDataset
        ├─ inputs  (N × input_dim)  ← StandardScaler normalized
        ├─ thermal (N × 1)          ← StandardScaler normalized
        ├─ stress  (N × 1)          ← StandardScaler normalized
        └─ em      (N × 1)          ← StandardScaler normalized
              │
              ▼
         DataLoader (train / val split)
              │
              ▼
         PINNTrainer ←── SelfCorrectionLoop (adaptive physics_weight)
              │
              ▼
         PINNModel.forward()
              │
              ▼
         physics_loss() → total_loss → backward() → Adam step
```
