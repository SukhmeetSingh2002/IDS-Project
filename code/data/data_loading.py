import pandas as pd

def load_dataset(training_file, testing_file):
    df_training = pd.read_csv(training_file)
    df_testing = pd.read_csv(testing_file)
    return df_training, df_testing

if __name__ == "__main__":
    df_training, df_testing = load_dataset(FILE_PATH_TRAINING, FILE_PATH_TESTING)
    
    print(df_training.head())
    print(df_testing.head())
    print(df_training.shape, df_testing.shape)

