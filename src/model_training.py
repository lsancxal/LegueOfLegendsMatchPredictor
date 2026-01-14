import torch
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=500, print_every=100):
    """Train the model and evaluate on test set.
    
    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        print_every: Print loss every N epochs
        
    Returns:
        Tuple of (test_predictions, test_accuracy)
    """
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_loss = running_loss / len(train_loader)

        if epoch % print_every == 0:
            print(f'Epoch [{epoch}], Train Loss: {train_loss:.4f}')

        # Evaluation phase
        model.eval()
        correct_test = 0
        total_test = 0
        all_test_outputs = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                all_test_outputs.append(outputs)
                preds = (outputs >= 0.5).float()
                correct_test += (preds == y_batch).sum().item()
                total_test += y_batch.size(0)

    train_accuracy = correct_train / total_train
    test_accuracy = correct_test / total_test
    print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    return torch.cat(all_test_outputs, dim=0), test_accuracy


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Create DataLoaders for training and testing.
    
    Args:
        X_train, y_train: Training data tensors
        X_test, y_test: Test data tensors
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
    
    return train_loader, test_loader
