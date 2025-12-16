import torch.optim as optim

import exercise_2 as e2
import exercise_3 as e3

L2_test_outputs = None
L2_accuracy = None


#EXERCISE 4: Model Optimization and Evaluation

def run_exercise_4(): 
    global optimizerL2, L2_test_outputs, L2_accuracy
    #Set Up the Optimizer with L2 Regularization:
    optimizerL2 = optim.SGD(e2.model.parameters(), lr=0.01, weight_decay=0.01)

    #Train the Model with L2 Regularization:
    L2_test_outputs, L2_accuracy = e3.train_model(optimizerL2, e2.model, e3.epochs, e3.train_loader, e3.test_loader)
  