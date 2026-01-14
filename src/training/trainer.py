"""Training logic for the model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        print_every: int = 100
    ):
        """
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer
            print_every: Print loss every N epochs
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_every = print_every
    
    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        
        return running_loss / len(train_loader), correct / total
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> tuple[torch.Tensor, float]:
        """Evaluate the model on test data.
        
        Returns:
            Tuple of (predictions, accuracy)
        """
        self.model.eval()
        correct = 0
        total = 0
        all_outputs = []
        
        for X_batch, y_batch in test_loader:
            outputs = self.model(X_batch)
            all_outputs.append(outputs)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        
        return torch.cat(all_outputs, dim=0), correct / total
    
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 500
    ) -> tuple[torch.Tensor, float]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            
        Returns:
            Tuple of (test_predictions, test_accuracy)
        """
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            if epoch % self.print_every == 0:
                print(f'Epoch [{epoch}], Train Loss: {train_loss:.4f}')
        
        test_preds, test_acc = self.evaluate(test_loader)
        print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        
        return test_preds, test_acc
