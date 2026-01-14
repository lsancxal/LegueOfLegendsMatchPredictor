"""Data loading and preprocessing utilities."""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .dataset import MatchDataset


def load_data(url: str) -> pd.DataFrame:
    """Load dataset from URL.
    
    Args:
        url: URL to the CSV dataset
        
    Returns:
        DataFrame with the loaded data
    """
    return pd.read_csv(url)


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = 'win',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess data for training.
    
    Args:
        df: Raw DataFrame
        target_column: Name of the target column
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as PyTorch tensors
    """
    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test


def create_data_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and testing.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = MatchDataset(X_train, y_train)
    test_dataset = MatchDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

    return train_loader, test_loader
