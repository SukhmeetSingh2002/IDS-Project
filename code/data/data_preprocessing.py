import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def remove_columns(df, columns):
    columns_to_remove = columns
    df.drop(columns_to_remove, axis=1, inplace=True)
    return df

def log_transform(df, columns, threshold=50):
    for column in columns:
        if df[column].nunique() > threshold:
            if (df[column] > 0).all():
                df[column] = np.log(df[column])
            elif (df[column] >= 0).all():
                df[column] = np.log(df[column] + 1)
            else:
                pass
    return df

def one_hot_encode(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df

def label_encode(df, columns):
    le = LabelEncoder()
    for column in columns:
        df[column] = le.fit_transform(df[column])
    return df





def preprocess_data(df, multi_class=False):
    columns_to_remove = ['id','proto','state','service','attack_cat']
    if multi_class:
        columns_to_remove.remove('attack_cat')
        columns_to_remove.append('label')
    remaining_columns = [column for column in df.columns if column not in columns_to_remove]


    df = remove_columns(df, columns_to_remove)
    df = log_transform(df, remaining_columns)
    # df = one_hot_encode(df, df.select_dtypes(include='object').columns)
    # df = label_encode(df, df.select_dtypes(include='object').columns)


    return df

def standardize_data(df_training, df_testing):
    standard_scaler = StandardScaler()

    for column in df_training.columns:
        if column != 'label' and column != 'attack_cat':
            standard_scaler.fit(df_training[column].values.reshape(-1,1))
            df_training[column] = standard_scaler.transform(df_training[column].values.reshape(-1,1))
            df_testing[column] = standard_scaler.transform(df_testing[column].values.reshape(-1,1))

    return df_training, df_testing

if __name__ == "__main__":
    FILE_PATH_TRAINING = "../dataset/UNSW_NB15_training-set.csv"
    FILE_PATH_TESTING = "../dataset/UNSW_NB15_testing-set.csv"
    
    df_training, df_testing = load_dataset(FILE_PATH_TRAINING, FILE_PATH_TESTING)
    
    def plot_normal_distribution(df):
        for column in df.columns:
            sns.displot(df[column])
            plt.show()
        
    print("--- Training data ---")
    plot_normal_distribution(df_training)

    print("--- Testing data ---")
    plot_normal_distribution(df_testing)