import torch.optim as optim
import matplotlib.pyplot as plt

import exercise_1 as e1
import exercise_2 as e2
import exercise_3 as e3

#EXERCISE 7: Hyperparameter Tuning
def main():
    #Define Learning Rates
    lr_list = [0.01, 0.05, 0.1]

    model_accuracy = []
    for i in range(len(lr_list)):
        #Reinitialize the Model for Each Learning Rate
        print(f'Training Model with Learning Rate: {lr_list[i]}')
        model = e2.LogisticRegressionModel(input_units=e2.input_dim, hidden_units=8, output_units=1)
        optimizer = optim.SGD(model.parameters(), lr = lr_list[i])
        #Train the Model for Each Learning Rate:
        model_test_outputs, accuracy = e3.train_model(optimizer, model, e3.epochs, e3.train_loader, e3.test_loader)
        model_accuracy.append(accuracy)

    #Evaluate and Compare:   
    plt.plot(lr_list, model_accuracy, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Model Test Accuracy vs Learning Rate')
    plt.show()

    max_accuracy = max(model_accuracy)
    max_accuracy_index = model_accuracy.index(max_accuracy)
    print(f'Max Accuracy: {max_accuracy:.4f} at Learning Rate: {lr_list[max_accuracy_index]}')  
