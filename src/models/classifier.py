"""Neural network model for binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    """Feedforward neural network for binary classification."""
    
    def __init__(self, input_dim: int, hidden_sizes: tuple = (16,), dropout_rate: float = 0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_sizes: Tuple of hidden layer sizes
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        layer_sizes = [input_dim] + list(hidden_sizes) + [1]
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layer = nn.Linear(in_size, out_size)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
                x = dropout(x)
            else:
                x = torch.sigmoid(layer(x))
        return x
