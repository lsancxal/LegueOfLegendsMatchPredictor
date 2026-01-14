"""Visualization utilities for model evaluation."""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Model predictions (probabilities)
    """
    y_pred_labels = (y_pred > 0.5).float()
    cm = confusion_matrix(y_true, y_pred_labels)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Loss', 'Win'], rotation=45)
    plt.yticks([0, 1], ['Loss', 'Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()


def plot_roc_curve(y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred: Model predictions (probabilities)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()


def plot_results(y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Plot all evaluation visualizations and print classification report.
    
    Args:
        y_true: True labels
        y_pred: Model predictions (probabilities)
    """
    y_pred_labels = (y_pred > 0.5).float()
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=['Loss', 'Win']))
    
    # Plot visualizations
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)
    plt.show()
