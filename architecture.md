# Network Architecture

## Overview

The PINN uses a shared encoder trunk with three domain-specific output heads.
All three physics outputs are computed in a single forward pass.

```
Input (batch × input\_dim)
        │
        ▼
┌───────────────────┐
│  Shared Encoder   │  Linear(input\_dim → 128) → ReLU
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

|Layer|In|Out|Activation|
|-|-|-|-|
|encoder\[0]|input\_dim (default 10)|128|ReLU|
|encoder\[1]|128|256|ReLU|
|encoder\[2]|256|128|ReLU|
|thermal\_head|128|1|none|
|stress\_head|128|1|none|
|em\_head|128|1|none|

Hidden dimensions are configurable via `hidden\_dims` at construction time.

## Class API

### `PINNModel(input\_dim, hidden\_dims, output\_dims)`

|Arg|Default|Description|
|-|-|-|
|`input\_dim`|`10`|Number of parametric input features|
|`hidden\_dims`|`\[128, 256, 128]`|Width of each encoder layer|
|`output\_dims`|`{'thermal':1,'stress':1,'EM':1}`|Output size per head|

**`forward(x)`** → `(thermal\_pred, stress\_pred, em\_pred)`

**`physics\_loss(..., physics\_weight)`** → `(total\_loss, data\_loss, physics\_loss, residuals\_dict)`

* `residuals\_dict` keys: `energy\_conservation`, `stress\_equilibrium`, `em\_smoothness`

### `PINNTrainer(model, learning\_rate, device)`

|Method|Description|
|-|-|
|`train\_epoch(dataloader, physics\_weight)`|One full training epoch, returns avg loss|
|`evaluate(dataloader)`|No-grad eval pass, returns avg loss|

Optimizer: Adam. Scheduler: StepLR (step=10, γ=0.9).

## Data Flow

```
CSV file
  └─► MultiPhysicsDataset
        ├─ inputs  (N × input\_dim)  ← StandardScaler normalized
        ├─ thermal (N × 1)          ← StandardScaler normalized
        ├─ stress  (N × 1)          ← StandardScaler normalized
        └─ em      (N × 1)          ← StandardScaler normalized
              │
              ▼
         DataLoader (train / val split)
              │
              ▼
         PINNTrainer ←── SelfCorrectionLoop (adaptive physics\_weight)
              │
              ▼
         PINNModel.forward()
              │
              ▼
         physics\_loss() → total\_loss → backward() → Adam step
```

