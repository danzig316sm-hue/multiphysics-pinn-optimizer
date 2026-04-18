# Multiphysics PINN Optimizer — Engineering Docs

Central index for all engineering documentation. Access any doc via the CLI viewer:

```bash
python docs_viewer.py list            # list all docs
python docs_viewer.py show <name>     # display a doc by name
python docs_viewer.py search <term>   # full-text search across all docs
```

---

## Documents

| File | Description |
|------|-------------|
| [index.md](index.md) | This file — navigation hub |
| [architecture.md](architecture.md) | Network architecture, layer shapes, data flow |
| [physics_equations.md](physics_equations.md) | Physics constraints, governing equations, derivations |
| [self_correction.md](self_correction.md) | Self-correcting loop design, parameters, tuning guide |
| [usage_guide.md](usage_guide.md) | End-to-end usage: data prep, training, inference |
| [troubleshooting.md](troubleshooting.md) | Common errors and fixes |

---

## Quick-Start

```python
from utils.data_loader import create_dataloader
from master_multi_physics_pipeline import run_pipeline

train_loader, val_loader, dataset = create_dataloader("data/my_data.csv")
model, history = run_pipeline(train_loader, val_loader, epochs=100)
```

The self-correcting loop runs automatically — no extra configuration needed.
