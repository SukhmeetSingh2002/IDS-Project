import matplotlib.pyplot as plt
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
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(y_true.unique()))
    plt.xticks(tick_marks, y_true.unique(), rotation=45)
    plt.yticks(tick_marks, y_true.unique())
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # save the plot
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')

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
