import torch
import torch.nn as nn
import torch.optim as optim

import exercise_1 as e1

input_dim = None
model = None
criterion = None
optimizer = None

#EXERCISE 2: Logistic Regression Model
# Define the Logistic Regression Mode
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_units=8, hidden_units=8, output_units=1):
        super(LogisticRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)       
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

def main():        
    global input_dim, model, criterion, optimizer
    #Initialize the Model, Loss Function, and Optimizer
    input_dim = e1.X_train.shape[1]
    model = LogisticRegressionModel(input_units=input_dim, hidden_units=8, output_units=1)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)