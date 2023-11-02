from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight

def split_dataset(df, with_validation):
    X = df.drop('label', axis=1)
    y = df['label']
    if with_validation:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


def oversample_data(X_train, y_train):
    oversampler = RandomOverSampler()
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled


def calculate_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return class_weights