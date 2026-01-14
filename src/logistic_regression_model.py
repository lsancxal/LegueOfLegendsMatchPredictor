import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_loading_and_preprocessing as dlp

input_dim = None
model = None
criterion = None
optimizer = None

#Logistic Regression Model
# Define the Logistic Regression Mode
class LogisticRegressionModel(nn.Module):
    #def __init__(self, input_units=8, hidden_units=16, output_units=1, dropout_rate=0.4):
    def __init__(self, Layers, dropout_rate=0.4):
        super(LogisticRegressionModel, self).__init__()
        #self.fc1 = nn.Linear(input_units, hidden_units)
        #self.dropout = nn.Dropout(dropout_rate)
        #self.fc2 = nn.Linear(hidden_units, output_units)
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, activation):
        #x = torch.relu(self.fc1(x))
        #x = self.dropout(x)
        #x = torch.sigmoid(self.fc2(x))
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
                activation = self.dropout(activation)
            else:
                activation = torch.sigmoid(linear_transform(activation))
        return activation

def run_logistic_regression_model():        
    global input_dim, model, criterion, optimizer
    #Initialize the Model, Loss Function, and Optimizer
    input_dim = dlp.X_train.shape[1]
    Layers = [input_dim, 16, 1]  
    model = LogisticRegressionModel(Layers, dropout_rate=0.3)  
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.1, weight_decay=0.01)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005) 