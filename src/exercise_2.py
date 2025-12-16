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
    def __init__(self, input_units=8, hidden_units=32, output_units=1):
        super(LogisticRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)       
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.fc5 = nn.Linear(hidden_units, output_units)
       # self.fc6 = nn.Linear(hidden_units, hidden_units)
       # self.fc7 = nn.Linear(hidden_units, hidden_units)
       # self.fc8 = nn.Linear(hidden_units, output_units)
       # self.fc9 = nn.Linear(hidden_units, hidden_units)
       # self.fc10 = nn.Linear(hidden_units, output_units)
       # self.fc11 = nn.Linear(hidden_units, hidden_units)
       # self.fc12 = nn.Linear(hidden_units, hidden_units)
       # self.fc13 = nn.Linear(hidden_units, hidden_units)
       # self.fc14 = nn.Linear(hidden_units, hidden_units)
       # self.fc15 = nn.Linear(hidden_units, hidden_units)
       # self.fc16 = nn.Linear(hidden_units, hidden_units)
       # self.fc17 = nn.Linear(hidden_units, hidden_units)
       # self.fc18 = nn.Linear(hidden_units, hidden_units)
       # self.fc19 = nn.Linear(hidden_units, hidden_units)
       # self.fc20 = nn.Linear(hidden_units, output_units)
       

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
       # x = torch.relu(self.fc6(x))
       # x = torch.relu(self.fc7(x))
       # x = torch.relu(self.fc8(x))
       # x = torch.relu(self.fc9(x))
       # x = torch.relu(self.fc10(x))
       # x = torch.relu(self.fc11(x))
       # x = torch.relu(self.fc12(x))
       # x = torch.relu(self.fc13(x))
       # x = torch.relu(self.fc14(x))
       # x = torch.relu(self.fc15(x))
       # x = torch.relu(self.fc16(x))
       # x = torch.relu(self.fc17(x))
       # x = torch.relu(self.fc18(x))
       # x = torch.relu(self.fc19(x))
       # x = torch.relu(self.fc20(x))
        x = torch.sigmoid(x)
        return x

def run_exercise_2():        
    global input_dim, model, criterion, optimizer
    #Initialize the Model, Loss Function, and Optimizer
    input_dim = e1.X_train.shape[1]
    model = LogisticRegressionModel(input_units=input_dim, hidden_units=32, output_units=1)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)