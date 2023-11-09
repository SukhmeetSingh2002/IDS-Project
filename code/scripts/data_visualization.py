import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_name} Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Training and Validation Loss')

    plt.tight_layout()
    plt.show()

    # save the plots
    plt.savefig(f'plots/{model_name}_training_history.png')

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

    # save the plot
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')


    plt.show()

def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # save the plot
    plt.savefig(f'plots/{model_name}_roc_curve.png')

def compare_predictions(y_true, y_pred_ann, y_pred_lstm, model_names):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred_ann, c='b', label=model_names[0])
    plt.scatter(y_true, y_pred_lstm, c='r', label=model_names[1])
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.legend(loc='upper left')
    plt.title('Predictions Comparison')

    plt.subplot(1, 2, 2)
    plt.hist([y_pred_ann, y_pred_lstm], bins=20, color=['b', 'r'], alpha=0.7, label=model_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Prediction Distributions')

    plt.tight_layout()
    plt.show()

    # save the plot
    plt.savefig(f'plots/predictions_comparison.png')

# plot classes distribution
def plot_classes_distribution(y,class_mapping, model_name):
    plt.figure(figsize=(12, 5))
    plt.hist(y, bins=20, color='b', alpha=0.7, label=class_mapping)
    plt.xticks(range(len(class_mapping)), class_mapping.keys(), rotation='vertical')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} Classes Distribution')

    plt.tight_layout()
    plt.show()

    # save the plot
    plt.savefig(f'plots/{model_name}_classes_distribution.png')

def plot_probabilities(y_prediction_array, true_label, name):
    predicted_label = np.argmax(y_prediction_array)

    plt.figure(figsize=(6, 3))
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), y_prediction_array, color="#777777")
    plt.ylim([0, 1])
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

    plt.show()

    # save the plot
    plt.savefig(f'plots/{name}_probabilities.png')


