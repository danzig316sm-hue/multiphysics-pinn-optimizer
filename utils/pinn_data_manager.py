"""
MULTIPHYSICS PINN OPTIMIZATION DATA MANAGER
Efficient storage for simulation results + training history + correction decisions

Integrates with: https://github.com/danzig316sm-hue/multiphysics-pinn-optimizer

Handles:
- CFD/Thermal/EM simulation results (large mesh data)
- PINN training history (metrics, losses, convergence)
- Self-correction loop decisions (the "story" of optimization)
- Model checkpoints (PyTorch .pt files)

Uses HDF5 for efficient hierarchical storage with compression
"""

import h5py
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle

class PINNOptimizationDataManager:
    """
    Manages all data from multiphysics PINN optimization runs
    
    Directory structure:
        runs/
        ├── run_20260330_143022/
        │   ├── simulation_results.h5    # CFD/Thermal/EM data
        │   ├── training_history.h5      # Loss curves, metrics
        │   ├── story.json                # Self-correction decisions
        │   ├── checkpoints/
        │   │   ├── best_model.pt
        │   │   ├── epoch_050.pt
        │   │   └── final_model.pt
        │   └── metadata.json            # Run parameters
        ├── run_20260330_163045/
        └── ...
    """
    
    def __init__(self, base_dir: str = "./pinn_optimization_runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_run_dir = None
        print(f"✓ PINN Data Manager initialized: {self.base_dir}")
    
    def start_new_run(self, 
                     run_name: Optional[str] = None,
                     parameters: Optional[Dict] = None) -> str:
        """
        Start a new optimization run
        
        Args:
            run_name: Optional custom name, otherwise timestamp-based
            parameters: Configuration dict (input_dim, epochs, lr, etc.)
        
        Returns:
            Run ID (directory name)
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        self.current_run_dir = self.base_dir / run_name
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.current_run_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "run_id": run_name,
            "start_time": datetime.now().isoformat(),
            "parameters": parameters or {}
        }
        
        with open(self.current_run_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Initialize story log
        story = {
            "run_id": run_name,
            "events": [],
            "decisions": []
        }
        
        with open(self.current_run_dir / "story.json", 'w') as f:
            json.dump(story, f, indent=2)
        
        print(f"✓ New run started: {run_name}")
        return run_name
    
    def save_simulation_results(self,
                                cfd_data: Optional[Dict[str, np.ndarray]] = None,
                                thermal_data: Optional[Dict[str, np.ndarray]] = None,
                                em_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Save simulation results (CFD, Thermal, EM)
        
        Args:
            cfd_data: Dict with keys like 'velocity', 'pressure', 'vorticity'
            thermal_data: Dict with keys like 'temperature', 'heat_flux'
            em_data: Dict with keys like 'E_field', 'B_field', 'current_density'
        
        Each array should be np.ndarray with shape matching mesh
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_new_run() first.")
        
        filepath = self.current_run_dir / "simulation_results.h5"
        
        with h5py.File(filepath, 'w') as f:
            # CFD Results
            if cfd_data:
                cfd_group = f.create_group('cfd')
                for key, data in cfd_data.items():
                    cfd_group.create_dataset(
                        key, 
                        data=data,
                        compression='gzip',
                        compression_opts=6,  # Good balance speed/size
                        shuffle=True  # Improves compression
                    )
                cfd_group.attrs['timestamp'] = datetime.now().isoformat()
            
            # Thermal Results
            if thermal_data:
                thermal_group = f.create_group('thermal')
                for key, data in thermal_data.items():
                    thermal_group.create_dataset(
                        key,
                        data=data,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True
                    )
                thermal_group.attrs['timestamp'] = datetime.now().isoformat()
            
            # EM Results
            if em_data:
                em_group = f.create_group('em')
                for key, data in em_data.items():
                    em_group.create_dataset(
                        key,
                        data=data,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True
                    )
                em_group.attrs['timestamp'] = datetime.now().isoformat()
        
        print(f"✓ Simulation results saved: {filepath.name}")
    
    def append_training_epoch(self,
                            epoch: int,
                            train_loss: float,
                            val_loss: float,
                            physics_loss: float,
                            data_loss: float,
                            learning_rate: float,
                            physics_weight: float,
                            grad_norm: Optional[float] = None,
                            additional_metrics: Optional[Dict[str, float]] = None):
        """
        Append training metrics for one epoch
        
        This builds up the training history incrementally
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_new_run() first.")
        
        filepath = self.current_run_dir / "training_history.h5"
        
        # Create file if doesn't exist
        if not filepath.exists():
            with h5py.File(filepath, 'w') as f:
                # Create extensible datasets (can grow)
                maxshape = (None,)  # Unlimited growth
                
                f.create_dataset('epoch', shape=(0,), maxshape=maxshape, dtype='i')
                f.create_dataset('train_loss', shape=(0,), maxshape=maxshape, dtype='f')
                f.create_dataset('val_loss', shape=(0,), maxshape=maxshape, dtype='f')
                f.create_dataset('physics_loss', shape=(0,), maxshape=maxshape, dtype='f')
                f.create_dataset('data_loss', shape=(0,), maxshape=maxshape, dtype='f')
                f.create_dataset('learning_rate', shape=(0,), maxshape=maxshape, dtype='f')
                f.create_dataset('physics_weight', shape=(0,), maxshape=maxshape, dtype='f')
                f.create_dataset('grad_norm', shape=(0,), maxshape=maxshape, dtype='f')
        
        # Append new data
        with h5py.File(filepath, 'a') as f:
            for name, value in [
                ('epoch', epoch),
                ('train_loss', train_loss),
                ('val_loss', val_loss),
                ('physics_loss', physics_loss),
                ('data_loss', data_loss),
                ('learning_rate', learning_rate),
                ('physics_weight', physics_weight),
                ('grad_norm', grad_norm or 0.0)
            ]:
                dataset = f[name]
                current_size = dataset.shape[0]
                dataset.resize(current_size + 1, axis=0)
                dataset[current_size] = value
            
            # Store additional metrics as attributes on last entry
            if additional_metrics:
                f.attrs[f'epoch_{epoch}_extra'] = json.dumps(additional_metrics)
    
    def log_story_event(self, 
                       event_type: str,
                       description: str,
                       epoch: Optional[int] = None,
                       data: Optional[Dict] = None):
        """
        Log a self-correction decision or important event
        
        This creates the "story" of the optimization
        
        Args:
            event_type: 'correction', 'checkpoint', 'convergence', 'error', etc.
            description: Human-readable explanation
            epoch: Which epoch this occurred (if applicable)
            data: Additional structured data
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_new_run() first.")
        
        filepath = self.current_run_dir / "story.json"
        
        # Load existing story
        with open(filepath, 'r') as f:
            story = json.load(f)
        
        # Add new event
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "description": description,
            "epoch": epoch,
            "data": data or {}
        }
        
        story["events"].append(event)
        
        # Save back
        with open(filepath, 'w') as f:
            json.dump(story, f, indent=2)
        
        print(f"  📖 Story: [{event_type}] {description}")
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None,
                       is_best: bool = False,
                       additional_state: Optional[Dict] = None):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
            epoch: Current epoch number
            is_best: If True, also save as 'best_model.pt'
            additional_state: Any extra data to save with checkpoint
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_new_run() first.")
        
        checkpoint_dir = self.current_run_dir / "checkpoints"
        
        # Build checkpoint dict
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if additional_state:
            checkpoint['additional_state'] = additional_state
        
        # Save epoch-specific checkpoint
        if epoch is not None:
            filename = f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_dir / filename)
            print(f"  💾 Checkpoint saved: {filename}")
        
        # Save as best if flagged
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ⭐ Best model saved: {best_path.name}")
            
            self.log_story_event(
                event_type="checkpoint",
                description=f"New best model at epoch {epoch}",
                epoch=epoch
            )
    
    def save_final_model(self, model: torch.nn.Module):
        """Save final trained model"""
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_new_run() first.")
        
        checkpoint_dir = self.current_run_dir / "checkpoints"
        final_path = checkpoint_dir / "final_model.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }, final_path)
        
        print(f"  ✅ Final model saved: {final_path.name}")
    
    def get_training_history(self) -> Dict[str, np.ndarray]:
        """
        Load training history for current run
        
        Returns dict with arrays for each metric
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run.")
        
        filepath = self.current_run_dir / "training_history.h5"
        
        if not filepath.exists():
            return {}
        
        history = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                history[key] = f[key][:]
        
        return history
    
    def get_story(self) -> Dict:
        """Load the story of optimization decisions"""
        if self.current_run_dir is None:
            raise RuntimeError("No active run.")
        
        filepath = self.current_run_dir / "story.json"
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def compress_run(self, run_dir: Optional[Path] = None):
        """
        Compress a completed run into a single archive
        
        This is for long-term storage / backups
        """
        import tarfile
        import gzip
        
        if run_dir is None:
            run_dir = self.current_run_dir
        
        if run_dir is None:
            raise RuntimeError("No run directory specified.")
        
        archive_path = run_dir.parent / f"{run_dir.name}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(run_dir, arcname=run_dir.name)
        
        print(f"✓ Run compressed: {archive_path}")
        print(f"  Original: {sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file()) / 1e6:.1f} MB")
        print(f"  Compressed: {archive_path.stat().st_size / 1e6:.1f} MB")
        
        return archive_path


# =============================================================================
# INTEGRATION EXAMPLE WITH YOUR PIPELINE
# =============================================================================

def integrated_pipeline_example():
    """
    Example showing how to integrate with your existing pipeline
    """
    # Initialize data manager
    data_mgr = PINNOptimizationDataManager()
    
    # Start new run
    run_id = data_mgr.start_new_run(
        parameters={
            'input_dim': 10,
            'epochs': 100,
            'learning_rate': 1e-3,
            'physics_weight_init': 0.1
        }
    )
    
    # ==== SIMULATION PHASE ====
    print("\n1. Running simulations...")
    
    # Placeholder: replace with actual simulation outputs
    cfd_results = {
        'velocity': np.random.randn(1000, 3),  # Your actual CFD velocity field
        'pressure': np.random.randn(1000),      # Your actual pressure field
    }
    
    thermal_results = {
        'temperature': np.random.randn(1000),   # Your actual temperature field
    }
    
    em_results = {
        'E_field': np.random.randn(1000, 3),    # Your actual E-field
    }
    
    data_mgr.save_simulation_results(
        cfd_data=cfd_results,
        thermal_data=thermal_results,
        em_data=em_results
    )
    
    # ==== TRAINING PHASE ====
    print("\n2. Training PINN...")
    
    # Simulate training loop
    for epoch in range(10):  # Your actual epoch loop
        # Your training step
        train_loss = 0.1 / (epoch + 1)
        val_loss = 0.12 / (epoch + 1)
        physics_loss = 0.05 / (epoch + 1)
        data_loss = 0.05 / (epoch + 1)
        
        # Log training metrics
        data_mgr.append_training_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            physics_loss=physics_loss,
            data_loss=data_loss,
            learning_rate=1e-3,
            physics_weight=0.1 * (1 + epoch * 0.1),
            grad_norm=0.5
        )
        
        # Self-correction decision
        if epoch % 5 == 0 and epoch > 0:
            data_mgr.log_story_event(
                event_type="correction",
                description=f"Increased physics weight due to high data loss",
                epoch=epoch,
                data={'old_weight': 0.1, 'new_weight': 0.2}
            )
        
        # Save checkpoint
        if epoch % 10 == 0:
            # data_mgr.save_checkpoint(model, optimizer, epoch)
            pass
    
    print("\n3. Run complete!")
    print(f"   Data saved to: {data_mgr.current_run_dir}")
    
    # ==== ANALYSIS ====
    history = data_mgr.get_training_history()
    story = data_mgr.get_story()
    
    print(f"\n   Training epochs: {len(history['epoch'])}")
    print(f"   Story events: {len(story['events'])}")
    
    # Compress for archival
    # data_mgr.compress_run()


if __name__ == "__main__":
    print("="*60)
    print("MULTIPHYSICS PINN OPTIMIZATION DATA MANAGER")
    print("="*60)
    
    integrated_pipeline_example()
