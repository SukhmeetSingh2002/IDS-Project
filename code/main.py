import numpy as np

from data.data_loading import load_dataset
from data.data_preprocessing import preprocess_data, standardize_data
from data.data_splitting import split_dataset, oversample_data

from models.models import (
    get_ann_model, train_ann_model, predict_ann, evaluate_model,
    get_lstm_model, train_lstm_model, predict_lstm
)

from scripts.data_visualization import *

if __name__ == "__main__":
    FILE_PATH_TRAINING = "../dataset/UNSW_NB15_training-set.csv"
    FILE_PATH_TESTING = "../dataset/UNSW_NB15_testing-set.csv"
    
    df_training, df_testing = load_dataset(FILE_PATH_TRAINING, FILE_PATH_TESTING)
    
    # Data preprocessing
    df_training = preprocess_data(df_training)
    df_testing = preprocess_data(df_testing)
    
    # Data standardization
    df_training, df_testing = standardize_data(df_training, df_testing)
    
    # Data splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df_training, with_validation=True)
    
    # Oversampling
    # X_train_oversampled, y_train_oversampled = oversample_data(X_train, y_train)
    
    # X_train, y_train = X_train_oversampled, y_train_oversampled
    
    # Build and train the ANN model
    ann_model = get_ann_model(X_train.shape[1])
    ann_model, history_ann = train_ann_model(ann_model, X_train, y_train, X_val, y_val)
    
    # Make predictions and evaluate
    y_prediction_ann = predict_ann(ann_model, X_test)
    evaluate_model(ann_model, X_test, y_test)
    
    # Build and train the LSTM model
    lstm_model = get_lstm_model((X_train.shape[1], 1))
    
    # Reshape test data for LSTM
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))

    lstm_model, history_lstm = train_lstm_model(lstm_model, X_train, y_train, X_val, y_val)
    
    
    # Make predictions and evaluate LSTM model
    y_prediction_lstm = predict_lstm(lstm_model, X_test)
    evaluate_model(lstm_model, X_test, y_test)

    # plot graphs

    # Plot training history
    plot_training_history(history_ann, 'ANN')
    plot_training_history(history_lstm, 'LSTM')

    y_prediction_ann = np.round(y_prediction_ann)
    y_prediction_lstm = np.round(y_prediction_lstm)

    # Plot confusion matrices
    plot_confusion_matrix(y_test, y_prediction_ann, 'ANN')
    plot_confusion_matrix(y_test, y_prediction_lstm, 'LSTM')

    # Plot ROC curves
    plot_roc_curve(y_test, y_prediction_ann, 'ANN')
    plot_roc_curve(y_test, y_prediction_lstm, 'LSTM')

    # Compare predictions
    compare_predictions(y_test, y_prediction_ann, y_prediction_lstm, ['ANN', 'LSTM'])




