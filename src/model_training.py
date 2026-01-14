import torch
from torch.utils.data import DataLoader, TensorDataset

import data_loading_and_preprocessing as dlp
import logistic_regression_model as lrm


#Set Number of Epochs:
epochs = 500
train_loader = None
test_loader = None
test_outputs = None

#Model Training
#Training Loop:
def train_model(optimizer, model,epochs, train_loader, test_loader):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = lrm.criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Print Loss Every 100 Epochs:
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}], Train Loss: {train_loss:.4f}')
        
        # Model Evaluation:
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        all_test_outputs = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                test_outputs = model(X_batch)
                all_test_outputs.append(test_outputs)
                loss = lrm.criterion(test_outputs, y_batch)
                test_loss += loss.item()
                preds = (test_outputs >= 0.5).float()
                correct_test += (preds == y_batch).sum().item()
                total_test += y_batch.size(0)

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

    # Calculate Accuracy:
    train_accuracy = correct_train / total_train
    test_accuracy = correct_test / total_test
    # Print Accuracy:
    print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return torch.cat(all_test_outputs, dim=0), test_accuracy

def run_train_model():
    global epochs, train_loader, test_loader, test_outputs
    # Create DataLoader for training and test sets
    train_dataset = TensorDataset(dlp.X_train, dlp.y_train)
    test_dataset = TensorDataset(dlp.X_test, dlp.y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
    test_outputs, _ = train_model(lrm.optimizer, lrm.model, epochs, train_loader, test_loader)