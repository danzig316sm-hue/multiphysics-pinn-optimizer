import torch
import torch.nn as nn
import torch.optim as optim

class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network for multi-physics optimization.
    Supports thermal, structural (stress), and electromagnetic predictions.
    """
    
    def __init__(self, input_dim=10, hidden_dims=[128, 256, 128], output_dims={'thermal': 1, 'stress': 1, 'EM': 1}):
        super(PINNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        
        # Shared encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads for different physics domains
        self.thermal_head = nn.Linear(hidden_dims[-1], output_dims.get('thermal', 1))
        self.stress_head = nn.Linear(hidden_dims[-1], output_dims.get('stress', 1))
        self.em_head = nn.Linear(hidden_dims[-1], output_dims.get('EM', 1))
    
    def forward(self, x):
        """
        Forward pass through PINN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            thermal_output, stress_output, em_output
        """
        h = self.encoder(x)
        thermal_output = self.thermal_head(h)
        stress_output = self.stress_head(h)
        em_output = self.em_head(h)
        return thermal_output, stress_output, em_output
    
    def physics_loss(self, thermal_pred, stress_pred, em_pred, 
                     thermal_true, stress_true, em_true,
                     physics_weight=0.1):
        """
        Compute combined data + physics loss.
        
        Args:
            *_pred: Model predictions
            *_true: Ground truth values
            physics_weight: Weight for physics constraints
            
        Returns:
            total_loss
        """
        # Data fidelity loss (MSE)
        mse_loss = nn.MSELoss()
        data_loss = (mse_loss(thermal_pred, thermal_true) +
                     mse_loss(stress_pred, stress_true) +
                     mse_loss(em_pred, em_true)) / 3.0
        
        # Physics constraints (simplified examples)
        # Energy conservation: thermal + EM should balance
        physics_constraint_1 = torch.mean(torch.abs(thermal_pred + em_pred * 0.1))
        
        # Stress equilibrium: stress should be non-negative
        physics_constraint_2 = torch.mean(torch.relu(-stress_pred))
        
        # EM field smoothness
        physics_constraint_3 = torch.mean(torch.abs(torch.diff(em_pred, dim=0)))
        
        physics_loss = (physics_constraint_1 + physics_constraint_2 + physics_constraint_3) / 3.0
        
        total_loss = data_loss + physics_weight * physics_loss
        return total_loss, data_loss, physics_loss


class PINNTrainer:
    """Training wrapper for PINN models."""
    
    def __init__(self, model, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
    
    def train_epoch(self, dataloader, physics_weight=0.1):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y_thermal, y_stress, y_em) in enumerate(dataloader):
            x = x.to(self.device)
            y_thermal = y_thermal.to(self.device)
            y_stress = y_stress.to(self.device)
            y_em = y_em.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            thermal_pred, stress_pred, em_pred = self.model(x)
            
            # Compute loss
            loss, data_loss, phys_loss = self.model.physics_loss(
                thermal_pred, stress_pred, em_pred,
                y_thermal, y_stress, y_em,
                physics_weight=physics_weight
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y_thermal, y_stress, y_em in dataloader:
                x = x.to(self.device)
                y_thermal = y_thermal.to(self.device)
                y_stress = y_stress.to(self.device)
                y_em = y_em.to(self.device)
                
                thermal_pred, stress_pred, em_pred = self.model(x)
                loss, _, _ = self.model.physics_loss(
                    thermal_pred, stress_pred, em_pred,
                    y_thermal, y_stress, y_em
                )
                total_loss += loss.item()
        
        return total_loss / len(dataloader)