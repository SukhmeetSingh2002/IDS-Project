from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from keras import regularizers
from keras.layers import BatchNormalization

import keras

from sklearn.ensemble import RandomForestClassifier
from tensorflow import cast as tf_cast
from tensorflow import float32 as tf_float32

from datetime import datetime


from tensorflow.keras import backend as K

def custom_weighted_loss(y_true, y_pred, class_weights):

    y_pred = tf_cast(y_pred, tf_float32)
    y_true = tf_cast(y_true, tf_float32)

    # Calculate the weighted loss
    weighted_loss = class_weights[0] * K.binary_crossentropy(y_true, y_pred)  # Class 0
    weighted_loss += class_weights[1] * K.binary_crossentropy(y_true, y_pred)  # Class 1
    return weighted_loss

# Define class weights
class_weights = {0: 10000.0, 1: 1.0}  # Adjust the weight for class 1 as needed

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

class MyRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=2, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.n_jobs = 70

    def fit(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=self.n_jobs)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

METRICS = [
      keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      keras.metrics.MeanSquaredError(name='Brier score'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


# def get_ann_model(input_dim):
#     model = Sequential()
#     model.add(Dense(100, input_dim=input_dim, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(75, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
    
#     # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.compile(loss=lambda y_true, y_pred: custom_weighted_loss(y_true, y_pred, class_weights), optimizer='adam', metrics=[f1_m, 'accuracy'])
#     return model

# multi-class 10 classes
def get_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(75, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy', precision_m, recall_m])
    return model

def train_ann_model(model, X_train, y_train, X_val, y_val, class_weights=None):
    if class_weights is None:
        history_ann = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=3000)
    else:
        history_ann = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=3000, class_weight=class_weights)
    return model, history_ann

def predict_ann(model, X_test):
    y_prediction_ann = model.predict(X_test)
    return y_prediction_ann


def get_lstm_model(input_shape):

    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True),activation='tanh')
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True),activation='tanh')
    model.add(Dropout(0.2))
    model.add(LSTM(10),activation='tanh')
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_lstm_model2(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(25, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(10, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val):
    history_lstm = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=512)
    return model, history_lstm

def predict_lstm(model, X_test):
    y_prediction_lstm = model.predict(X_test)
    return y_prediction_lstm


# Saving and loading models

def save_ann_model(model, accuracy):
    model_file = f"models/saved_models/model_ann_{get_current_datetime()}_{accuracy:.3f}.keras"
    model.save(model_file)

def save_lstm_model(model, accuracy):
    model_file = f"models/saved_models/model_lstm_{get_current_datetime()}_{accuracy:.3f}.keras"
    model.save(model_file)

def load_ann_model(model_file):
    return load_model(model_file)

def load_lstm_model(model_file):
    return load_model(model_file)

def evaluate_model(model, X_test, y_test):
    accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}%")

    accuracy = 10.123456789


    # save the model to disk
    if model.layers[0].name == 'lstm':
        save_lstm_model(model, accuracy)
    else:
        save_ann_model(model, accuracy)
