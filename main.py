"""Entry point for the League of Legends Match Predictor."""

import random
import numpy as np
import torch

from src.config import Config
from src.data import load_data, preprocess_data, create_data_loaders
from src.models import BinaryClassifier
from src.training import Trainer
from src.visualization import plot_results


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config: Config = None) -> float:
    """Run the full training pipeline.
    
    Args:
        config: Configuration object (uses defaults if None)
        
    Returns:
        Test accuracy
    """
    if config is None:
        config = Config()
    
    set_seed(config.seed)
    print("Starting the League of Legends Match Predictor...")
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data(config.dataset_url)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=config.batch_size
    )
    
    # Create model
    print("Creating model...")
    input_dim = X_train.shape[1]
    model = BinaryClassifier(
        input_dim=input_dim,
        hidden_sizes=config.hidden_sizes,
        dropout_rate=config.dropout_rate
    )
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # Train model
    print("Training model...")
    trainer = Trainer(model, criterion, optimizer, print_every=config.print_every)
    test_preds, test_acc = trainer.fit(train_loader, test_loader, epochs=config.epochs)
    
    # Visualize results
    print("Generating visualizations...")
    plot_results(y_test, test_preds)
    
    print("Completed successfully!")
    return test_acc


if __name__ == "__main__":
    main()
