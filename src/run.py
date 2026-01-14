import random
import numpy as np
import torch

from data_loading_and_preprocessing import load_and_preprocess_data
from logistic_regression_model import create_model
from model_training import create_data_loaders, train_model
from plot_confusion_matrix import plot_results

# Configuration
SEED = 1
DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"

# Hyperparameters
EPOCHS = 500
BATCH_SIZE = 64
HIDDEN_SIZES = [16]
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
MOMENTUM = 0.1


def main():
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Starting the League of Legends Match Predictor...")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATASET_URL)
    
    # Create model
    print("Creating model...")
    input_dim = X_train.shape[1]
    model, criterion, optimizer = create_model(
        input_dim,
        hidden_sizes=HIDDEN_SIZES,
        dropout_rate=DROPOUT_RATE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE
    )
    
    # Train model
    print("Training model...")
    test_predictions, test_accuracy = train_model(
        model, criterion, optimizer, train_loader, test_loader, epochs=EPOCHS
    )
    
    # Plot results
    print("Plotting results...")
    plot_results(test_predictions, y_test)
    
    print("League of Legends Match Predictor completed successfully.")


if __name__ == "__main__":
    main()
