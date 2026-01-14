import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionModel(nn.Module):
    """Neural network for binary classification with configurable hidden layers."""
    
    def __init__(self, layer_sizes, dropout_rate=0.3):
        """
        Args:
            layer_sizes: List of layer sizes, e.g. [input_dim, 16, 1]
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layer = nn.Linear(in_size, out_size)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
                x = dropout(x)
            else:
                x = torch.sigmoid(layer(x))
        return x


def create_model(input_dim, hidden_sizes=[16], dropout_rate=0.3, lr=0.001, weight_decay=0.005, momentum=0.1):
    """Create model, criterion, and optimizer.
    
    Args:
        input_dim: Number of input features
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout probability
        lr: Learning rate
        weight_decay: L2 regularization strength
        
    Returns:
        Tuple of (model, criterion, optimizer)
    """
    layer_sizes = [input_dim] + hidden_sizes + [1]
    model = LogisticRegressionModel(layer_sizes, dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    return model, criterion, optimizer
