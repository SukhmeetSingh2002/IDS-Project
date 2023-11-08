import numpy as np

from data.data_loading import load_dataset
from data.data_preprocessing import preprocess_data, standardize_data
from data.data_splitting import split_dataset, oversample_data, split_dataset_testing, upsample_data, calculate_class_weights, downsample_data, label_encode, to_categorical_label_encode

import warnings
warnings.filterwarnings("ignore")

from models.models import (
    get_ann_model, train_ann_model, predict_ann, evaluate_model,
    get_lstm_model, train_lstm_model, predict_lstm, MyRandomForestClassifier, get_lstm_model2, load_ann_model
)

from scripts.data_visualization import *

if __name__ == "__main__":
    FILE_PATH_TRAINING = "../dataset/UNSW_NB15_training-set.csv"
    FILE_PATH_TESTING = "../dataset/UNSW_NB15_testing-set.csv"
    
    df_training, df_testing = load_dataset(FILE_PATH_TRAINING, FILE_PATH_TESTING)
    
    # Data preprocessing
    df_training = preprocess_data(df_training, multi_class=True)
    df_testing = preprocess_data(df_testing, multi_class=True)
    
    # Data standardization
    df_training, df_testing = standardize_data(df_training, df_testing)
    
    # label encode
    df_training, df_testing = label_encode(df_training, df_testing, multi_class=True)

    # Data splitting
    X_train, X_val, y_train, y_val = split_dataset(df_training, multi_class=True)
    X_test, y_test = split_dataset_testing(df_testing, multi_class=True)

    # Oversampling
    # X_train_oversampled, y_train_oversampled = oversample_data(X_train, y_train)

    X_train_oversampled, y_train_oversampled = upsample_data(X_train, y_train, multi_class=True)
    # X_train_oversampled, y_train_oversampled = downsample_data(X_train, y_train)
    
    X_train, y_train = X_train_oversampled, y_train_oversampled


    # to categorical
    y_train = to_categorical_label_encode(y_train)
    y_val = to_categorical_label_encode(y_val)
    y_test_one_hot = to_categorical_label_encode(y_test)

    # print shape of data
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")


    # class weights
    # class_weights = calculate_class_weights(y_train)

    print(f"Value counts of y_train: {np.unique(y_train, return_counts=True)}")
    print(f"Value counts of y_val: {np.unique(y_val, return_counts=True)}")
    print(f"Value counts of y_test: {np.unique(y_test, return_counts=True)}")
    # print(f"Class weights: {class_weights}")
    
    # Build and train the ANN model
    ann_model = get_ann_model(X_train.shape[1])
    ann_model, history_ann = train_ann_model(ann_model, X_train, y_train, X_val, y_val)

    # LOAD MODEL
    # ann_model = load_ann_model("models/saved_models/model_ann_20231103-112820_0.887.keras")
    
    # Make predictions and evaluate
    y_prediction_ann = predict_ann(ann_model, X_test)
    evaluate_model(ann_model, X_test, y_test_one_hot)


    ############ Random Forest ############
    rf = MyRandomForestClassifier(n_estimators=100, max_depth=100, random_state=42)
    rf.fit(X_train, y_train)
    y_prediction_rf = rf.predict(X_test)
    y_prediction_rf = np.argmax(y_prediction_rf, axis=1)

    print("y_prediction_rf",y_prediction_rf.shape)
    print("y_test",y_test.shape)

    # calulate accuracy using y_prediction_rf
    from sklearn.metrics import accuracy_score
    print("accuracy",accuracy_score(y_test, y_prediction_rf))




    ############ Random Forest ############
    
    # Build and train the LSTM model
    lstm_model = get_lstm_model2((X_train.shape[1], 1))
    
    # Reshape test data for LSTM
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))

    lstm_model, history_lstm = train_lstm_model(lstm_model, X_train, y_train, X_val, y_val)

    # lstm_model = load_ann_model("models/saved_models/model_lstm_20231103-113019_0.872.keras")
    
    
    # Make predictions and evaluate LSTM model
    y_prediction_lstm = predict_lstm(lstm_model, X_test)
    evaluate_model(lstm_model, X_test, y_test)

    # plot graphs

    # Plot training history
    plot_training_history(history_ann, 'ANN')
    plot_training_history(history_lstm, 'LSTM')

    y_prediction_ann = np.round(y_prediction_ann)
    y_prediction_lstm = np.round(y_prediction_lstm)

    # plot y_prediction_ann and y_test
    import matplotlib.pyplot as plt
    plt.plot(y_prediction_ann, label='y_prediction_ann')
    plt.plot(y_test, label='y_test')
    plt.legend()

    # save the plot
    plt.savefig(f'plots/compare_y_prediction_ann_lstm.png')
    plt.show()

    # Plot confusion matrices
    plot_confusion_matrix(y_test, y_prediction_ann, 'ANN')
    plot_confusion_matrix(y_test, y_prediction_lstm, 'LSTM')

    # Plot ROC curves
    plot_roc_curve(y_test, y_prediction_ann, 'ANN')
    plot_roc_curve(y_test, y_prediction_lstm, 'LSTM')

    # Compare predictions
    compare_predictions(y_test, y_prediction_ann, y_prediction_lstm, ['ANN', 'LSTM'])




