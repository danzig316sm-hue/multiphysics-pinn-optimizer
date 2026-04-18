---
name: ml-ai-development
description: >
  Use this skill for ANY ML/AI development task in the Mobius-Nova platform —
  PyTorch model architecture, training loops, GPU optimization, Bayesian
  optimization, active learning, surrogate modeling, data pipeline design,
  model deployment, hyperparameter tuning, loss function design, dataset
  preparation, or debugging training issues. Trigger whenever the user mentions
  PyTorch, training loop, model architecture, GPU memory, CUDA optimization,
  Bayesian optimizer, Latin hypercube sampling, surrogate model, active learning,
  data loader, checkpoint, learning rate, batch size, overfitting, validation loss,
  cuML, cuDNN, TensorRT, NVIDIA Modulus, or any ML engineering task. Also trigger
  for Supermemory integration, FlowState session management, TurboQuant compression,
  or design genome database queries. This is the primary skill for all AI/ML
  engineering work on the platform.
---

# ML/AI Development Skill

## Project Stack
```
PyTorch 2.0+          — Core ML framework
Python 3.11/3.12      — Stable (3.14 on local, use 3.11 for Colab)
Google Colab          — GPU training (T4 free, A100 pro)
Juno Notebook         — Local development (Windows 11)
GitHub                — Version control
TurboQuant            — KV cache compression (pip install turboquant)
Supermemory           — Design genome database
FlowState V1.1        — Session state continuity (MCP)
cuML                  — GPU-accelerated ML (RAPIDS)
```

---

## Environment Setup

### Local (Windows 11 + Juno)
```python
# Check environment
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# TurboQuant status
try:
    from turboquant import TurboQuantCache
    print("TurboQuant: ACTIVE — 4x KV compression enabled")
except ImportError:
    print("TurboQuant: NOT INSTALLED — pip install turboquant")
```

### Google Colab Setup
```python
# Run at start of every Colab session
!pip install turboquant featool pyfemm pyDOE2 cadquery -q

# Mount GitHub repo
from google.colab import drive
import subprocess
subprocess.run(['git', 'clone', 
    'https://github.com/danzig316sm-hue/multiphysics-pinn-optimizer'])

import sys
sys.path.insert(0, '/content/multiphysics-pinn-optimizer')

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

---

## Training Best Practices

### DataLoader Pattern For Simulation Data
```python
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class GeneratorDesignDataset(Dataset):
    """
    Dataset for PINN training from simulation results.
    X = geometry features (36-dim)
    Y = performance outputs (7-dim)
    """
    def __init__(self, designs: list, augment: bool = False):
        self.X = []
        self.Y = []
        
        for d in designs:
            # Build feature vector
            x = (
                d["bezier_curve1"] +
                d["bezier_curve2"] + 
                d["bezier_curve3"] +
                [d["ratio_parameter"] / 100.0,
                 1.0 if d.get("multimaterial") else 0.0,
                 d.get("rated_speed_rpm", 150.0) / 200.0]
            )
            while len(x) < 36:
                x.append(0.0)
            
            # Build target vector — MUST match output head layout
            y = [
                (d["efficiency_pct"] - 90.0) / 10.0,      # em[0]
                (d["cogging_torque_nm"] - 5.0) / 40.0,    # em[1]
                (d["brmin_tesla"] - 0.1) / 0.5,           # em[2]
                (d["magnet_mass_kg"] - 5.0) / 15.0,       # em[3]
                (d["max_winding_temp_c"] - 60.0) / 150.0, # thermal[0]
                (d["max_magnet_temp_c"] - 30.0) / 80.0,   # thermal[1]
                (d["max_stress_pa"] - 10e6) / 200e6,      # structural[0]
            ]
            
            self.X.append(x)
            self.Y.append(y)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_dataloader(designs, batch_size=32, val_split=0.2):
    """Split designs into train/val and create DataLoaders."""
    n = len(designs)
    n_val = int(n * val_split)
    
    train_data = GeneratorDesignDataset(designs[n_val:])
    val_data   = GeneratorDesignDataset(designs[:n_val])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                              shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size,
                              shuffle=False, pin_memory=True)
    
    return train_loader, val_loader
```

---

## Bayesian Optimization For Design Space Search

```python
# pip install botorch ax-platform
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

def bayesian_optimize_generator(
    evaluate_fn,           # function: geometry_vector → performance_dict
    n_initial: int = 50,   # initial random samples
    n_iterations: int = 200,  # BO iterations
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Bayesian optimization over generator design space.
    
    Objective: minimize magnet_mass subject to constraints:
      - efficiency >= 95%
      - brmin >= 0.3 T
      - cogging <= 25 N·m
    """
    bounds = torch.zeros(2, 36, device=device)
    bounds[1] = 1.0  # all features normalized 0-1
    
    # Initial random sampling (Latin Hypercube)
    from pyDOE2 import lhs
    X_init = torch.tensor(
        lhs(36, samples=n_initial), 
        dtype=torch.float32, device=device
    )
    
    # Evaluate initial designs
    Y_init = []
    for x in X_init:
        result = evaluate_fn(x.cpu().numpy())
        # Objective: maximize negative magnet mass (minimize mass)
        obj = -result.get("magnet_mass_kg", 15.0)
        # Apply constraint penalties
        if result.get("efficiency_pct", 0) < 95.0:
            obj -= 10.0
        if result.get("brmin_tesla", 0) < 0.3:
            obj -= 10.0
        Y_init.append([obj])
    
    Y_init = torch.tensor(Y_init, dtype=torch.float32, device=device)
    
    # BO loop
    best_designs = []
    for i in range(n_iterations):
        # Fit GP surrogate
        gp = SingleTaskGP(X_init, Y_init)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # Optimize acquisition function
        acqf = qExpectedImprovement(gp, best_f=Y_init.max())
        candidates, _ = optimize_acqf(
            acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=50
        )
        
        # Evaluate candidate
        result = evaluate_fn(candidates[0].cpu().numpy())
        obj = -result.get("magnet_mass_kg", 15.0)
        
        # Update dataset
        X_init = torch.cat([X_init, candidates])
        Y_init = torch.cat([Y_init, torch.tensor([[obj]], device=device)])
        
        if i % 20 == 0:
            best_mass = -Y_init.max().item()
            print(f"[BO iter {i}] Best magnet mass: {best_mass:.2f} kg")
        
        best_designs.append({
            "iteration": i,
            "geometry_vector": candidates[0].cpu().numpy().tolist(),
            "result": result,
        })
    
    return best_designs
```

---

## cuML Integration (GPU-Accelerated ML)

```python
# pip install cuml-cu12 (for CUDA 12)
# Or use RAPIDS container on Colab

def gpu_accelerated_surrogate_screening(
    candidates: np.ndarray,
    trained_surrogate,
    n_top: int = 100
):
    """
    Use cuML for fast pre-screening before full PINN evaluation.
    Orders of magnitude faster than sklearn for large candidate sets.
    """
    try:
        from cuml.ensemble import RandomForestRegressor as cuRF
        from cuml.preprocessing import StandardScaler as cuScaler
        
        scaler = cuScaler()
        X_scaled = scaler.fit_transform(candidates)
        
        # Fast GPU inference
        predictions = trained_surrogate.predict(X_scaled)
        
    except ImportError:
        # Fallback to sklearn
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(candidates)
        predictions = trained_surrogate.predict(X_scaled)
    
    # Return indices of top N candidates
    top_indices = np.argsort(predictions)[:n_top]
    return candidates[top_indices]
```

---

## Supermemory — Design Genome Database

```python
from supermemory import Supermemory
import os

client = Supermemory(api_key=os.environ.get("SUPERMEMORY_API_KEY"))

def save_design_to_genome(design_hash: str, result: dict):
    """Save evaluated design to persistent design genome."""
    client.memories.add(
        content=f"""
        Design {design_hash}:
        Efficiency: {result.get('efficiency_pct', 'N/A')}%
        Cogging: {result.get('cogging_torque_nm', 'N/A')} Nm
        Magnet mass: {result.get('magnet_mass_kg', 'N/A')} kg
        Brmin: {result.get('brmin_tesla', 'N/A')} T
        Solver: {result.get('solver', 'unknown')}
        Trust score: {result.get('trust_score', 'N/A')}
        """,
        container_tags=["generator-designs", "mobius-nova"]
    )

def search_similar_designs(query: str, n: int = 10):
    """Semantic search over design genome."""
    results = client.memories.search(
        query=query,
        container_tags=["generator-designs"],
        limit=n
    )
    return results
```

---

## GPU Memory Management

```python
def optimize_batch_size_for_gpu(
    model,
    input_dim: int = 36,
    start_batch: int = 512,
    device: str = "cuda"
) -> int:
    """
    Find maximum batch size that fits in GPU memory.
    Useful for Colab T4 (15GB) with TurboQuant active.
    """
    model = model.to(device)
    batch_size = start_batch
    
    while batch_size > 1:
        try:
            dummy = torch.randn(batch_size, input_dim, device=device)
            _ = model(dummy)
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError:  # OOM
            batch_size //= 2
            torch.cuda.empty_cache()
    
    return 1


def log_gpu_memory(label: str = ""):
    """Quick GPU memory check during training."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU {label}] {allocated:.2f}GB allocated / "
              f"{reserved:.2f}GB reserved / {total:.2f}GB total")
```

---

## Checkpoint Management

```python
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "turboquant_bits": getattr(model, "bits", 4),
    }, path)

def load_checkpoint(model, optimizer, path):
    if not Path(path).exists():
        return 0, []
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["history"]
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Colab disconnects mid-training | Save checkpoint every 10 epochs |
| Python 3.14 PyTorch incompatibility | Use Python 3.11 venv as fallback |
| cuML not available | Fall back to sklearn, log warning |
| Supermemory API key missing | Check SUPERMEMORY_API_KEY env var |
| TurboQuant OOM on 3-bit | Switch to 4-bit or reduce batch size |
| GitHub push fails from Colab | Use personal access token |
| FEATool import error | Running in stub mode — expected |

---

## Colab Workflow (Standard Session)

```python
# 1. Setup (run once per session)
!git clone https://github.com/danzig316sm-hue/multiphysics-pinn-optimizer
!pip install turboquant supermemory pyDOE2 -q

# 2. Load latest checkpoint
from master_pipeline_v2 import run_training
from utils.turboquant_wrapper import compression_report
compression_report()  # verify TurboQuant active

# 3. Load QBlade datasets
from qblade_automation import ingest_qblade_rotor_performance
data = ingest_qblade_rotor_performance("qblade_data.csv")

# 4. Train
train_loader, val_loader = create_dataloader(designs)
model, history = run_training(
    train_loader, val_loader,
    epochs=100,
    turboquant_bits=3,  # max compression on T4
    checkpoint_dir="checkpoints"
)

# 5. Push results
!git add . && git commit -m "Training run $(date)" && git push
```

---

## NVIDIA Modulus (Future Phase)

When Inception GPU credits arrive, migrate PINN to Modulus for:
- Native PDE constraint handling
- Built-in magnetostatics and heat equation modules  
- Multi-GPU training support
- NGC model registry for surrogate storage

```python
# Future: modulus.sym for physics constraints
# from modulus.sym.eq.pdes.maxwell import MaxwellEquation
# from modulus.sym.eq.pdes.diffusion import Diffusion
```
