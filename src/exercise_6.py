
import torch

import exercise_1 as e1
import exercise_2 as e2
import exercise_3 as e3


#EXERCISE 6: Model Saving and Loading
def evaluate_model(model, test_loader):
    # Ensure the loaded model is in evaluation mode
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_test_outputs = []
    test_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            test_outputs = model(X_batch)
            all_test_outputs.append(test_outputs)
            loss = e2.criterion(test_outputs, y_batch)
            test_loss += loss.item()
            preds = (test_outputs >= 0.5).float()
            correct_test += (preds == y_batch).sum().item()
            total_test += y_batch.size(0)
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    test_accuracy = correct_test / total_test
    # Print Accuracy:
    print(f'Loaded models Test Accuracy: {test_accuracy:.4f}')
    return torch.cat(all_test_outputs, dim=0)


def main():
    # Save the model
    torch.save(e2.model.state_dict(), 'lol_model.pth')

    # Load the model
    new_model = e2.LogisticRegressionModel(input_units=e2.input_dim, hidden_units=8, output_units=1)
    new_model.load_state_dict(torch.load('lol_model.pth'))

    loaded_model_outputs = evaluate_model(new_model, e3.test_loader)
