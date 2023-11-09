from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from keras import regularizers
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.initializers import he_normal

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Activation, Add, Input, SeparableConv1D, AveragePooling1D, Multiply
from keras.models import Model
from keras.layers import Reshape


import keras
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from tensorflow import cast as tf_cast
from tensorflow import float32 as tf_float32

from sklearn.metrics import classification_report

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
    def __init__(self, n_estimators=100, max_depth=2, random_state=0, class_weight=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.class_weights = class_weight
        self.n_jobs = 70

    def fit(self, X_train, y_train):
        if self.class_weights is None:
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=self.n_jobs)
        else:
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=self.n_jobs, class_weight=self.class_weights)
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
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=[f1_m, 'accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=[f1_m, 'accuracy'])
    return model


def get_cnn_model(input_shape):
    num_classes = 10
    input_shape = (input_shape, 1)
    input_tensor = Input(shape=input_shape)

    # Initial Convolution
    x = Conv1D(64, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2(0.01))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    num_filters = 64
    for _ in range(3):
        x = _residual_block(x, num_filters, l2_reg=0.01)
        num_filters *= 2

    # Final layers
    x = GlobalAveragePooling1D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(input_tensor, output_tensor)

    # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy', f1_m])
    # model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy', f1_m])

    # Print the model summary
    print(model.summary())

    return model

    

def _residual_block(input_tensor, num_filters, l2_reg=0.01):
    if input_tensor.shape[-1] != num_filters:
        # Apply 1x1 convolution to match dimensions
        input_tensor = Conv1D(num_filters, kernel_size=1, strides=1, padding='same', kernel_regularizer=l2(l2_reg))(input_tensor)

    x = Conv1D(num_filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_reg))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(num_filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def get_cnn_model2(input_shape):
    input_shape = (input_shape, 1)

    model = Sequential()
    model.add(Conv1D(128, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2(0.01), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # max pooling and then flatten
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy', f1_m])
    # model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy', f1_m])

    # Print the model summary
    print(model.summary())

    return model

# Convolutional Neural Network with lstm
def get_cnn_model3(input_shape):
    
    input_shape = (input_shape, 1)
    num_labels = 10
    
    inp = Input(shape=input_shape)
    x = BatchNormalization()(inp)
    x = Dropout(0.1)(x)
    x = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x = AveragePooling1D(pool_size=2)(x)
    xs = x
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x = Multiply()([x, xs])
    x = MaxPooling1D(pool_size=4, strides=2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1024, kernel_initializer=he_normal(), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, kernel_initializer=he_normal(), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_labels, kernel_initializer=he_normal(), activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m])

    # Print the model summary
    print(model.summary())

    return model 
    

def train_ann_model(model, X_train, y_train, X_val, y_val, class_weights=None):
    if class_weights is None:
        history_ann = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=2048)
    else:
        history_ann = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=2048, class_weight=class_weights)
    return model, history_ann

def predict_ann(model, X_test):
    y_prediction_ann = model.predict(X_test)
    return y_prediction_ann


# def get_lstm_model(input_shape):

#     model = Sequential()
#     model.add(LSTM(100, input_shape=input_shape, return_sequences=True),activation='tanh')
#     model.add(Dropout(0.2))
#     model.add(LSTM(50, return_sequences=True),activation='tanh')
#     model.add(Dropout(0.2))
#     model.add(LSTM(10),activation='tanh')
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return model


def get_lstm_model(input_shape):
    input_shape = (input_shape, 1)
    model = Sequential()
    model.add(LSTM(150, input_shape=input_shape, return_sequences=True, activation='tanh', ))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True, activation='tanh', ))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='tanh', ))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])

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
    accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}%")

    accuracy = accuracy[1]


    # save the model to disk
    if model.layers[0].name == 'lstm':
        save_lstm_model(model, accuracy)
    else:
        save_ann_model(model, accuracy)

    return accuracy

def save_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'reports/{model_name}_classification_report.csv', index=True)