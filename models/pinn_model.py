import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))  # Using tanh activation
        x = self.layers[-1](x)  # Output layer
        return x

# Physics-informed loss function
def physics_informed_loss(model, x, y_true):
    model.eval()
    y_pred = model(x)
    loss_data = nn.MSELoss()(y_pred, y_true)

    # Physics-informed terms (example: gradients)
    # You would add specific PDE residuals based on your physics constraints here
    # loss_physics = ...

    loss = loss_data  # + loss_physics if defined
    return loss

# Example usage
# input_dim = number of input features
# output_dim = number of outputs (temperature, stress, etc.)
# hidden_layers = list of hidden layer sizes
# model = PINN(input_dim, output_dim, hidden_layers)