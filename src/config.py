"""Configuration and hyperparameters for the model."""

from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for all hyperparameters."""
    
    # Data
    dataset_url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"
    test_size: float = 0.2
    random_state: int = 42
    
    # Model architecture
    hidden_sizes: tuple = (16,)
    dropout_rate: float = 0.3
    
    # Training
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    momentum: float = 0.1
    
    # Reproducibility
    seed: int = 1
    
    # Logging
    print_every: int = 100
