from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer

from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight

from keras.utils import to_categorical

import pandas as pd
import numpy as np


def split_dataset(df, multi_class=False):
    if multi_class:
        X = df.drop(['attack_cat'], axis=1)
        y = df['attack_cat']
    else:
        X = df.drop('label', axis=1)
        y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def label_encode(df_training, df_testing, multi_class=False):
    if multi_class:
        le = LabelEncoder()
        df_training['attack_cat'] = le.fit_transform(df_training['attack_cat'])
        df_testing['attack_cat'] = le.transform(df_testing['attack_cat'])
    else:
        pass
    return df_training, df_testing


def split_dataset_testing(df, multi_class=False):
    if multi_class:
        X = df.drop(['attack_cat'], axis=1)
        y = df['attack_cat']
    else:
        X = df.drop('label', axis=1)
        y = df['label']
    return X, y

def to_categorical_label_encode(y):
    y = to_categorical(y, num_classes=y.nunique())
    return y


def oversample_data(X_train, y_train):
    oversampler = RandomOverSampler()
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled


def calculate_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return class_weights


from sklearn.utils import resample


def upsample_data(X_train, y_train, multi_class=False):


    # concatenate our training data back together
    df = pd.concat([X_train, y_train], axis=1)


    if multi_class:
        # separate minority and majority classes
        majority_class = df.attack_cat.value_counts().idxmax()

        # separate minority and majority classes
        majority = df[df.attack_cat==majority_class]

         # list to collect upsampled minority classes
        upsampled_minorities = []

        # iterate over all classes and upsample minority classes
        for class_label in df.attack_cat.unique():
            if class_label != majority_class:
                minority = df[df.attack_cat==class_label]
                minority_upsampled = resample(minority,
                                            replace=True,  # sample with replacement
                                            n_samples=len(majority),  # match number in majority class
                                            random_state=27)  # reproducible results
                upsampled_minorities.append(minority_upsampled)

        # combine majority and upsampled minority
        upsampled = pd.concat([majority] + upsampled_minorities)

        # check new class counts
        print(upsampled.attack_cat.value_counts())

        y_train = upsampled['attack_cat']
        X_train = upsampled.drop('attack_cat', axis=1)
        
        return X_train, y_train
        

    # separate minority and majority classes
    majority = df[df.label==1]
    minority = df[df.label==0]

    # upsample minority
    minority_upsampled = resample(minority,
                              replace=True, # sample with replacement
                              n_samples=len(majority), # match number in majority class
                              random_state=27) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([majority, minority_upsampled])

    # check new class counts
    print(upsampled.label.value_counts())

    y_train = upsampled['label']
    X_train = upsampled.drop('label', axis=1)

    return X_train, y_train

def downsample_data(X_train, y_train):

    # concatenate our training data back together
    df = pd.concat([X_train, y_train], axis=1)

    # separate minority and majority classes
    majority = df[df.label==1]
    minority = df[df.label==0]

    # downsample majority
    majority_downsampled = resample(majority,
                                replace=False, # sample without replacement
                                n_samples=len(minority), # match minority n
                                random_state=27) # reproducible results

    # combine minority and downsampled majority
    downsampled = pd.concat([minority, majority_downsampled])

    # check new class counts
    print(downsampled.label.value_counts())

    y_train = downsampled['label']
    X_train = downsampled.drop('label', axis=1)

    return X_train, y_train