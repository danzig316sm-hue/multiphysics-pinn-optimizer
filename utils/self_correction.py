"""
Self-Correction Loop for Physics-Informed Neural Network Training

Adaptively adjusts physics loss weight during training based on validation performance.
Generates "story" of optimization decisions for analysis.

Usage:
    from utils.self_correction import SelfCorrectionLoop
    
    loop = SelfCorrectionLoop(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        physics_weight_init=0.1,
        checkpoint_dir="checkpoints",
        verbose=True
    )
    
    history = loop.run(epochs=100)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


class SelfCorrectionLoop:
    """
    Adaptive training loop with physics weight self-correction
    
    Monitors validation loss and adjusts physics loss weight to balance
    data fitting vs physics constraint satisfaction.
    
    Attributes:
        trainer: PINNTrainer instance
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        physics_weight: Current physics loss weight (adaptive)
        best_val_loss: Best validation loss seen
        checkpoint_dir: Where to save best model
        verbose: Print training progress
    """
    
    def __init__(self,
                 trainer,
                 train_loader,
                 val_loader,
                 physics_weight_init: float = 0.1,
                 checkpoint_dir: str = "checkpoints",
                 verbose: bool = True):
        """
        Initialize self-correction loop
        
        Args:
            trainer: PINNTrainer with model and optimizer
            train_loader: Training data
            val_loader: Validation data
            physics_weight_init: Initial physics loss weight
            checkpoint_dir: Directory for checkpoints
            verbose: Print epoch-level progress
        """
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.physics_weight = physics_weight_init
        self.physics_weight_min = 0.01
        self.physics_weight_max = 10.0
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.patience_threshold = 10  # Epochs to wait before correction
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.verbose = verbose
        self.history = []
        self.correction_log = []
        
        # Save initial state
        self.best_model_state = None
        
    def should_correct(self) -> bool:
        """
        Determine if physics weight should be adjusted
        
        Returns:
            True if correction needed
        """
        return self.patience_counter >= self.patience_threshold
    
    def adjust_physics_weight(self, epoch: int, val_loss: float, physics_loss: float, data_loss: float):
        """
        Adjust physics weight based on loss imbalance
        
        Strategy:
        - If physics_loss >> data_loss: Decrease physics weight
        - If data_loss >> physics_loss: Increase physics weight
        - If val_loss plateaued: Increase physics weight (enforce constraints)
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            physics_loss: Current physics loss component
            data_loss: Current data loss component
        """
        old_weight = self.physics_weight
        correction_reason = ""
        
        # Compute loss ratio
        if data_loss > 1e-8:
            loss_ratio = physics_loss / data_loss
        else:
            loss_ratio = 1.0
        
        # Correction logic
        if loss_ratio > 10.0:
            # Physics loss dominates - reduce weight
            self.physics_weight *= 0.8
            correction_reason = f"Physics loss >> data loss (ratio: {loss_ratio:.2f})"
        
        elif loss_ratio < 0.1:
            # Data loss dominates - increase weight
            self.physics_weight *= 1.2
            correction_reason = f"Data loss >> physics loss (ratio: {loss_ratio:.2f})"
        
        elif self.patience_counter >= self.patience_threshold:
            # Val loss plateaued - enforce physics more
            self.physics_weight *= 1.5
            correction_reason = f"Val loss plateaued for {self.patience_counter} epochs"
        
        # Clamp to bounds
        self.physics_weight = np.clip(
            self.physics_weight,
            self.physics_weight_min,
            self.physics_weight_max
        )
        
        # Log correction
        if abs(self.physics_weight - old_weight) > 1e-6:
            self.correction_log.append({
                'epoch': epoch,
                'old_weight': float(old_weight),
                'new_weight': float(self.physics_weight),
                'reason': correction_reason,
                'val_loss': float(val_loss),
                'physics_loss': float(physics_loss),
                'data_loss': float(data_loss),
                'timestamp': datetime.now().isoformat()
            })
            
            if self.verbose:
                print(f"\n  🔧 Self-Correction:")
                print(f"     Physics weight: {old_weight:.4f} → {self.physics_weight:.4f}")
                print(f"     Reason: {correction_reason}\n")
            
            # Reset patience after correction
            self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict with train_loss, physics_loss, data_loss
        """
        self.trainer.model.train()
        total_loss = 0.0
        total_physics = 0.0
        total_data = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.trainer.device)
            targets = targets.to(self.trainer.device)
            
            # Forward pass
            outputs = self.trainer.model(inputs)
            
            # Data loss (MSE)
            data_loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # Physics loss (placeholder - override in your implementation)
            physics_loss = self.trainer.compute_physics_loss(inputs, outputs)
            
            # Combined loss
            loss = data_loss + self.physics_weight * physics_loss
            
            # Backward pass
            self.trainer.optimizer.zero_grad()
            loss.backward()
            self.trainer.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_physics += physics_loss.item()
            total_data += data_loss.item()
        
        n_batches = len(self.train_loader)
        return {
            'train_loss': total_loss / n_batches,
            'physics_loss': total_physics / n_batches,
            'data_loss': total_data / n_batches
        }
    
    def validate(self) -> float:
        """
        Compute validation loss
        
        Returns:
            Validation loss (float)
        """
        self.trainer.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.trainer.device)
                targets = targets.to(self.trainer.device)
                
                outputs = self.trainer.model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: If True, save as best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'physics_weight': self.physics_weight,
            'best_val_loss': self.best_val_loss,
            'correction_log': self.correction_log
        }
        
        # Save epoch checkpoint
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_state = checkpoint['model_state_dict']
            
            if self.verbose:
                print(f"  ⭐ New best model saved (val_loss: {self.best_val_loss:.6f})")
    
    def load_best(self):
        """Load best model weights"""
        if self.best_model_state is not None:
            self.trainer.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"\n✓ Best model restored (epoch {self.best_epoch})")
    
    def run(self, epochs: int) -> List[Dict]:
        """
        Run training loop with self-correction
        
        Args:
            epochs: Total number of epochs
        
        Returns:
            Training history (list of dicts with metrics per epoch)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Self-Correcting PINN Training")
            print(f"  Epochs: {epochs}")
            print(f"  Initial physics weight: {self.physics_weight:.4f}")
            print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Self-correction logic
            if self.should_correct():
                self.adjust_physics_weight(
                    epoch,
                    val_loss,
                    train_metrics['physics_loss'],
                    train_metrics['data_loss']
                )
            
            # Record history
            epoch_record = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_loss,
                'physics_loss': train_metrics['physics_loss'],
                'data_loss': train_metrics['data_loss'],
                'physics_weight': self.physics_weight,
                'learning_rate': self.trainer.optimizer.param_groups[0]['lr']
            }
            self.history.append(epoch_record)
            
            # Print progress
            if self.verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train: {train_metrics['train_loss']:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"Physics: {train_metrics['physics_loss']:.6f} | "
                      f"Data: {train_metrics['data_loss']:.6f} | "
                      f"λ: {self.physics_weight:.4f}")
            
            # Save periodic checkpoint
            if epoch % 50 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save correction log
        log_path = self.checkpoint_dir / "correction_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.correction_log, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"  Best val loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})")
            print(f"  Total corrections: {len(self.correction_log)}")
            print(f"  Final physics weight: {self.physics_weight:.4f}")
            print(f"{'='*60}\n")
        
        return self.history


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Self-correction loop module loaded successfully")
    print("This module is typically imported, not run directly")
