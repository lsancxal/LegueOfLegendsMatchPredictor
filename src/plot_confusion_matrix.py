import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def plot_results(y_pred, y_true):
    """Plot confusion matrix and ROC curve.
    
    Args:
        y_pred: Model predictions (probabilities)
        y_true: True labels
    """
    y_pred_labels = (y_pred > 0.5).float()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Loss', 'Win'], rotation=45)
    plt.yticks([0, 1], ['Loss', 'Win'])

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=['Loss', 'Win']))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
