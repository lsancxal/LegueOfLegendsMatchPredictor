import pandas as pd
import matplotlib.pyplot as plt

import exercise_1 as e1
import exercise_2 as e2

#EXERCISE 8: Feature Importance

def run_exercise_8():
    #Extracting Model Weights
    model_weights = e2.model.fc1.weight.data.abs().mean(dim=0).numpy()
    feature_names = e1.X.columns
    #Creating a DataFrame
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model_weights}) 
    #Sorting and Plotting Feature Importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()
