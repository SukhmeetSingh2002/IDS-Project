from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight

import pandas as pd
import numpy as np

def split_dataset(df):
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def split_dataset_testing(df):
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y


def oversample_data(X_train, y_train):
    oversampler = RandomOverSampler()
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled


def calculate_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return class_weights


from sklearn.utils import resample


def upsample_data(X_train, y_train):

    # concatenate our training data back together
    df = pd.concat([X_train, y_train], axis=1)

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