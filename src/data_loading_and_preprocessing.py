import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

X_train = None
X_test = None
y_train = None
y_test = None
X = None

#Data Loading and Preprocessing
def run_data_loading_and_preprocessing(url):
    global X_train, X_test, y_train, y_test, input_dim, X
    #Load the dataset
    lolDf = pd.read_csv(url)

    #Split data into features and targe
    X = lolDf.drop('win', axis=1)
    y = lolDf['win'] 

    #Split the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)