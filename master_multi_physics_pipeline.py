"""
Master Multi-Physics Pipeline — UPGRADED
Replaces master_multi_physics_pipeline.py stubs with real solvers.

Changes from original:
  - run_cfd_simulation()     → FEAToolSolver.run_cfd()
  - run_thermal_simulation() → FEAToolSolver.run_thermal()
  - run_em_simulation()      → FEAToolSolver.run_electromagnetic()
  - PINNModel                → TurboQuantPINN (drop-in, 4x memory compression)
  - NEW: SolidWorksVerification parallel channel
  - NEW: GeometrySpec structured input
  - Existing SelfCorrectionLoop unchanged — still works the same way

Install new deps:
  pip install turboquant featool
  pip install pyfemm  (optional EM fallback)

Quick start:
  python master_pipeline_v2.py
"""

import sys
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Solver imports
# ---------------------------------------------------------------------------
from solvers.base_solver import GeometrySpec
from solvers.featool_solver import FEAToolSolver
from solvers.sw_verification import SolidWorksVerification

# ---------------------------------------------------------------------------
# PINN imports — upgraded with TurboQuant
# ---------------------------------------------------------------------------
from utils.turboquant_wrapper import TurboQuantPINN, MultiPhysicsLoss, compression_report

# ---------------------------------------------------------------------------
# Keep your existing utilities unchanged
# ---------------------------------------------------------------------------
try:
    from models.pinn_model import PINNTrainer      # your original trainer
    from utils.self_correction import SelfCorrectionLoop  # your original loop
    from utils.data_loader import create_dataloader
    ORIGINAL_UTILS_AVAILABLE = True
except ImportError:
    ORIGINAL_UTILS_AVAILABLE = False
    print("Note: Original utils not found — running standalone mode")


# ---------------------------------------------------------------------------
# Simulation functions — REAL implementations replacing stubs
# ---------------------------------------------------------------------------

_solver = None  # singleton — init once

def get_solver() -> FEAToolSolver:
    global _solver
    if _solver is None:
        _solver = FEAToolSolver(mesh_density="medium", verbose=True)
    return _solver


def run_cfd_simulation(geometry: GeometrySpec = None, **kwargs) -> dict:
    """
    Replaces the stub. Real OpenFOAM CFD via FEATool.
    Falls back to physics-consistent stub if FEATool not installed.
    """
    if geometry is None:
        geometry = _default_geometry()
    result = get_solver().run_cfd(geometry, kwargs.get("flow_conditions"))
    return result


def run_thermal_simulation(geometry: GeometrySpec = None, **kwargs) -> dict:
    """
    Replaces the stub. Real FEniCS thermal FEA via FEATool.
    Target: Twinding < 180°C, Tmagnet < 60°C
    """
    if geometry is None:
        geometry = _default_geometry()
    result = get_solver().run_thermal(geometry, kwargs.get("heat_sources"))
    return result


def run_em_simulation(geometry: GeometrySpec = None, **kwargs) -> dict:
    """
    Replaces the stub. Real magnetostatic FEA via FEATool.
    Target: η ≥ 95%, cogging < 25 N·m, Brmin > 0.3 T
    """
    if geometry is None:
        geometry = _default_geometry()
    result = get_solver().run_electromagnetic(geometry, kwargs.get("materials"))
    return result


def run_simulations(geometry: GeometrySpec = None) -> dict:
    """Run all three domains. Drop-in for the original run_simulations()."""
    print("Starting Multi-Physics Simulation Pipeline...")
    if geometry is None:
        geometry = _default_geometry()

    solver = get_solver()
    results = solver.run_all(geometry)

    print(f"\nSimulation pipeline completed.")
    print(f"  Design hash: {geometry.design_hash()}")
    print(f"  Total time: {results['total_time_s']:.1f}s")
    print(f"  Solver: {results['solver']}")
    return results


# ---------------------------------------------------------------------------
# Training — upgraded PINN with TurboQuant
# ---------------------------------------------------------------------------

def run_training(
    train_loader,
    val_loader,
    *,
    input_dim: int = 36,       # 33 Bézier + ratio + 2 material flags
    epochs: int = 100,
    learning_rate: float = 1e-3,
    physics_weight_init: float = 0.1,
    checkpoint_dir: str = "checkpoints",
    turboquant_bits: int = 4,  # 3=5x compression, 4=4x compression
    verbose: bool = True,
):
    """
    Build TurboQuantPINN, wrap in SelfCorrectionLoop, train.
    Drop-in upgrade for the original run_training().

    New arg: turboquant_bits — set to 3 for max compression on Colab T4
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Show compression status
    compression_report()

    # Build upgraded model
    model = TurboQuantPINN(
        input_dim=input_dim,
        bits=turboquant_bits,
        device=device,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Use your existing trainer if available, otherwise basic training loop
    if ORIGINAL_UTILS_AVAILABLE:
        trainer = PINNTrainer(model, learning_rate=learning_rate, device=device)
        loop = SelfCorrectionLoop(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            physics_weight_init=physics_weight_init,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
        )
        history = loop.run(epochs=epochs)
        loop.load_best()
    else:
        history = _basic_training_loop(
            model, train_loader, val_loader,
            epochs=epochs, lr=learning_rate,
            checkpoint_dir=checkpoint_dir, verbose=verbose,
        )

    return model, history


def _basic_training_loop(
    model, train_loader, val_loader, epochs, lr, checkpoint_dir, verbose
):
    """Fallback training loop when original utils not available."""
    Path(checkpoint_dir).mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MultiPhysicsLoss(physics_weight=0.1)
    history = []
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            preds = model(x)
            # Build target dict from y tensor
            targets = {
                "em": y[:, :4],
                "thermal": y[:, 4:6],
                "structural": y[:, 6:7],
            }
            loss, breakdown = loss_fn(preds, targets, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                preds = model(x)
                targets = {"em": y[:, :4], "thermal": y[:, 4:6], "structural": y[:, 6:7]}
                loss, _ = loss_fn(preds, targets, x)
                val_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        record = {"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val}
        history.append(record)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f}")

    return history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    train_loader=None,
    val_loader=None,
    epochs: int = 100,
    geometry: GeometrySpec = None,
    enable_sw_verification: bool = True,
    turboquant_bits: int = 4,
):
    """
    Full pipeline: geometry → simulations → PINN training → SW verification queue.

    Args:
        train_loader, val_loader: PyTorch DataLoaders (optional)
        epochs: Training epochs
        geometry: GeometrySpec for this run (uses baseline if None)
        enable_sw_verification: Flag top designs for SolidWorks check
        turboquant_bits: 3 or 4 (3 = more compression, 4 = more accuracy)
    """
    if geometry is None:
        geometry = _default_geometry()

    print(f"\n{'='*65}")
    print(f"  MULTIPHYSICS PINN OPTIMIZER — v2")
    print(f"  Design: {geometry.design_hash()}")
    print(f"  TurboQuant: {turboquant_bits}-bit KV compression")
    print(f"{'='*65}\n")

    # Run simulations
    sim_results = run_simulations(geometry)

    # Optional SW verification queue
    if enable_sw_verification:
        sw = SolidWorksVerification(
            watch_folder="sw_verification",
            mode="manual",
            verbose=True,
        )
        design_hash = sw.flag_for_verification(
            geometry=geometry,
            pipeline_results={
                "structural": sim_results.get("structural"),
                "thermal": sim_results.get("thermal"),
                "electromagnetic": sim_results.get("electromagnetic"),
            },
            priority="normal",
            notes="auto-flagged by pipeline",
        )
        print(f"\n[Pipeline] Design queued for SW verification: {design_hash}")
        cutover = sw.check_cutover_readiness()
        print(f"[Pipeline] SW cutover readiness: {cutover['message']}")

    # PINN training
    if train_loader is not None and val_loader is not None:
        model, history = run_training(
            train_loader,
            val_loader,
            epochs=epochs,
            turboquant_bits=turboquant_bits,
        )
        print(f"\n[Pipeline] Training complete.")
        print(f"[Pipeline] Final val loss: {history[-1]['val_loss']:.6f}")
        return model, history, sim_results

    print("\n[Pipeline] No dataloaders — skipping PINN training.")
    return None, [], sim_results


# ---------------------------------------------------------------------------
# Default baseline geometry — Bergey 15-kW NREL baseline
# ---------------------------------------------------------------------------

def _default_geometry() -> GeometrySpec:
    """
    Baseline arc-shaped pole from NREL paper.
    60-50 slot-pole, 150 RPM, 15 kW.
    """
    return GeometrySpec(
        bezier_curve1=[0.5] * 6,          # flat air gap (no profiling)
        bezier_curve2=[0.3, 0.4, 0.5, 0.5, 0.4, 0.3],   # arc shape
        bezier_curve3=[0.2, 0.25, 0.3, 0.3, 0.25, 0.2],  # core boundary
        ratio_parameter=80.0,
        magnet_grade="NdFeB_bonded_75vol",
        rated_speed_rpm=150.0,
        rated_power_w=15000.0,
    )


# ---------------------------------------------------------------------------
# Colab / GPU detection (keep from original)
# ---------------------------------------------------------------------------

if "google.colab" in sys.modules:
    if torch.cuda.is_available():
        print("Google Colab: GPU runtime detected.")
        compression_report()
    else:
        print("Google Colab: CPU runtime — consider switching to GPU runtime.")
        print("Runtime → Change runtime type → T4 GPU")


if __name__ == "__main__":
    print("Running pipeline with baseline geometry (stub mode if FEATool not installed)...")
    model, history, sim_results = run_pipeline(
        enable_sw_verification=True,
        turboquant_bits=4,
    )
    print("\nDone. Check sw_verification/pending_export/ for SolidWorks session files.")
