import random
import numpy as np
import torch

import data_loading_and_preprocessing as dlp
import logistic_regression_model as lrm
import model_training as mt
import plot_confusion_matrix as pcm

# Set random seed for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"

def main():
    print("Starting the League of Legends Match Predictor...")
    print("Starting data loading and preprocessing")
    dlp.run_data_loading_and_preprocessing(DATASET_URL)
    print("Starting logistic regression model")
    lrm.run_logistic_regression_model()
    print("Starting model training")
    mt.run_train_model()
    print("Starting plot confusion matrix")
    pcm.run_plot_confusion_matrix()
    print("League of Legends Match Predictor completed successfully.")

main()