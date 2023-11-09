import numpy as np

from data.data_loading import load_dataset
from data.data_preprocessing import preprocess_data, standardize_data
from data.data_splitting import split_dataset, oversample_data, split_dataset_testing, upsample_data, calculate_class_weights, downsample_data, label_encode, to_categorical_label_encode

import warnings
warnings.filterwarnings("ignore")

from models.models import *
from scripts.data_visualization import *


# Machine Learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

# funtion to run neural network models
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, y_test_one_hot , train_model, name, class_weights=None):

    
    # # Reshape test data for LSTM
    if name == 'LSTM':
        X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
        X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
        

    model, history = train_model(model, X_train, y_train, X_val, y_val)

    # Make predictions and evaluate
    y_prediction = predict_ann(model, X_test)
    accuracy = evaluate_model(model, X_test, y_test_one_hot)

    print(f'{name} Accuracy: {accuracy}')

    # Plot graphs
    plot_training_history(history, name)

    # Plot probabilities
    for i in range(56, 60):
        plot_probabilities(y_prediction[i], y_test[i], name=f'prediction_{i}_{name}')

    # Convert predictions to labels
    y_prediction = np.argmax(y_prediction, axis=1)

    # Plot confusion matrices
    plot_confusion_matrix(y_test, y_prediction, name)

    # Plot ROC curves
    # plot_roc_curve(y_test, y_prediction, name)

    # save classification report to file
    save_classification_report(y_test, y_prediction, name)

    # save accuracy to file
    with open('accuracy.txt', 'a') as f:
        f.write(f'{name} Accuracy: {accuracy}\n')

    


# funtion to run machine learning models
def train_ml_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    y_prediction = np.argmax(y_prediction, axis=1)

    accuracy = accuracy_score(y_test, y_prediction)

    print(f'{name} Accuracy: {accuracy}')

    # Plot confusion matrices
    plot_confusion_matrix(y_test, y_prediction, name)

    # Plot ROC curves
    # plot_roc_curve(y_test, y_prediction, name)

    # save classification report to file
    save_classification_report(y_test, y_prediction, name)

    # save accuracy to file
    with open('accuracy.txt', 'a') as f:
        f.write(f'{name} Accuracy: {accuracy}\n')



if __name__ == "__main__":
    FILE_PATH_TRAINING = "../dataset/UNSW_NB15_training-set.csv"
    FILE_PATH_TESTING = "../dataset/UNSW_NB15_testing-set.csv"

    with open('accuracy.txt', 'a') as f:
        f.write('')
        f.write('Multi-class classification\n\n')
    
    df_training, df_testing = load_dataset(FILE_PATH_TRAINING, FILE_PATH_TESTING)
    
    # Data preprocessing
    df_training = preprocess_data(df_training, multi_class=True)
    df_testing = preprocess_data(df_testing, multi_class=True)

    # different columns and remove them appropriately
    # c1=df_training.columns
    # c2=df_testing.columns
    # # columns in training but not in testing
    # difference = set(c1) - set(c2)
    # df_training = df_training.drop(difference, axis=1)
    # # columns in testing but not in training
    # difference = set(c2) - set(c1)
    # df_testing = df_testing.drop(difference, axis=1)

    
    # Data standardization
    df_training, df_testing = standardize_data(df_training, df_testing)
    
    # label encode
    df_training, df_testing, class_mapping = label_encode(df_training, df_testing, multi_class=True)


    # Data splitting
    X_train, X_val, y_train, y_val = split_dataset(df_training, multi_class=True)
    X_test, y_test = split_dataset_testing(df_testing, multi_class=True)

    # Oversampling
    # X_train_oversampled, y_train_oversampled = oversample_data(X_train, y_train)

    # X_train_oversampled, y_train_oversampled = upsample_data(X_train, y_train, multi_class=True)
    # X_train_oversampled, y_train_oversampled = downsample_data(X_train, y_train)
    
    # X_train, y_train = X_train_oversampled, y_train_oversampled


    y_train_original = y_train
    # plot class distribution
    plot_classes_distribution(y_test,class_mapping, 'Test')
    plot_classes_distribution(y_train_original,class_mapping, 'Train')
    plot_classes_distribution(y_val,class_mapping, 'Validation')

    # X_test=X_test[60_000:]
    # y_test=y_test[60_000:]
    
    # to categorical
    y_train = to_categorical_label_encode(y_train)
    y_val = to_categorical_label_encode(y_val)
    y_test_one_hot = to_categorical_label_encode(y_test)


    # Nerual Network models: Storing the get model funtion and train model funtion in a dictionary
    nn_models = {
        'ANN': (get_ann_model, train_ann_model),
        'LSTM': (get_lstm_model, train_lstm_model),
        'CNN': (get_cnn_model2, train_ann_model)
    }


    # class weights
    class_weights = None
    # class_weights = calculate_class_weights(y_train)
    # class_weights = {i: weight for i, weight in enumerate(class_weights)}

    
    # Train and evaluate neural network models
    for name, (get_model, train_model) in nn_models.items():
        try:
            print(f'Training {name} model...')
            model = get_model(X_train.shape[1])
            train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test ,y_test_one_hot, train_model, name)
        except Exception as e:
            print(f'Error while training {name} model.')
            print(e)
            print()



    
    ml_models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=70),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }

    # Train and evaluate machine learning models
    for name, model in ml_models.items():
        try:
            print(f'Training {name} model...')
            train_ml_model(model, X_train, y_train, X_test, y_test, name)
        except Exception as e:
            print(f'Error while training {name} model.')
            print(e)
            print()



