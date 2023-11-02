from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

from datetime import datetime

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def get_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_ann_model(model, X_train, y_train, X_val, y_val):
    history_ann = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=2048)
    return model, history_ann

def predict_ann(model, X_test):
    y_prediction_ann = model.predict(X_test)
    return y_prediction_ann


def get_lstm_model(input_shape):

    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val):
    history_lstm = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=2048)
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
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # save the model to disk
    if model.layers[0].name == 'lstm':
        save_lstm_model(model, accuracy)
    else:
        save_ann_model(model, accuracy)
