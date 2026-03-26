"""
Master Multi-Physics Simulation Pipeline.

Orchestrates CFD, Thermal, and EM simulations followed by PINN training
with an adaptive self-correcting loop.
"""

import sys
import subprocess

import torch

from models.pinn_model import PINNModel, PINNTrainer
from utils.self_correction import SelfCorrectionLoop


# ---------------------------------------------------------------------------
# Simulation stubs (replace with real solvers)
# ---------------------------------------------------------------------------

def run_cfd_simulation():
    """Run CFD simulation (placeholder)."""
    print("Running CFD simulation...")
    # subprocess.run(['cfd_solver', 'input_file'], check=True)


def run_thermal_simulation():
    """Run Thermal simulation (placeholder)."""
    print("Running Thermal simulation...")
    # subprocess.run(['thermal_solver', 'input_file'], check=True)


def run_em_simulation():
    """Run EM simulation (placeholder)."""
    print("Running EM simulation...")
    # subprocess.run(['em_solver', 'input_file'], check=True)


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

def run_simulations():
    """Execute all three simulation domains sequentially."""
    print("Starting Multi-Physics Simulation Pipeline...")
    run_cfd_simulation()
    run_thermal_simulation()
    run_em_simulation()
    print("Simulation pipeline completed.")


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
    """
    Build a PINN, wrap it in a SelfCorrectionLoop, and train.

    Args:
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data.
        input_dim: Number of input features.
        epochs: Total training epochs.
        learning_rate: Initial Adam learning rate.
        physics_weight_init: Starting physics loss weight.
        checkpoint_dir: Where to save best-model checkpoints.
        verbose: Print epoch-level progress.

    Returns:
        Tuple of (trained PINNModel, training history list).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PINNModel(input_dim=input_dim)
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

    # Restore best weights before returning
    loop.load_best()

    return model, history


def run_pipeline(train_loader=None, val_loader=None, epochs: int = 100):
    """
    Full pipeline: simulations → PINN training with self-correction.

    If train_loader/val_loader are None the simulation stubs run but
    PINN training is skipped (useful for testing simulation flow only).
    """
    run_simulations()

    if train_loader is not None and val_loader is not None:
        model, history = run_training(train_loader, val_loader, epochs=epochs)
        print(f"Training complete. Final val loss: {history[-1]['val_loss']:.6f}")
        return model, history

    print("No dataloaders provided — skipping PINN training.")
    return None, []


# ---------------------------------------------------------------------------
# Google Colab / GPU detection
# ---------------------------------------------------------------------------

if "google.colab" in sys.modules:
    if torch.cuda.is_available():
        print("Google Colab: GPU runtime detected.")
    else:
        print("Google Colab: using CPU runtime.")

if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    from utils.data_loader import create_dataloader

    print("No data file specified — running pipeline simulation stubs only.")
    run_pipeline()
