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
    def __init__(self, input_units=8, hidden_units=16, output_units=1, dropout_rate=0.4):
        super(LogisticRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def run_exercise_2():        
    global input_dim, model, criterion, optimizer
    #Initialize the Model, Loss Function, and Optimizer
    input_dim = e1.X_train.shape[1]
    model = LogisticRegressionModel(input_units=input_dim, hidden_units=16, output_units=1, dropout_rate=0.4)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)