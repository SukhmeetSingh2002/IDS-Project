import pandas as pd

def load_dataset(training_file, testing_file):
    df_training = pd.read_csv(training_file, index_col=0)
    df_testing = pd.read_csv(testing_file, index_col=0)
    return df_training, df_testing

# plot id vs label
def plot_id_vs_label(df, name):
    df.plot(x='id', y='label', kind='scatter')
    plt.show()

    # save
    plt.savefig(f"id_vs_label_{name}.png")

if __name__ == "__main__":
    FILE_PATH_TRAINING = "../dataset/UNSW_NB15_training-set.csv"
    FILE_PATH_TESTING = "../dataset/UNSW_NB15_testing-set.csv"
    df_training, df_testing = load_dataset(FILE_PATH_TRAINING, FILE_PATH_TESTING)
    
    print(df_training.head())
    # print(df_testing.head())
    # print(df_training.shape, df_testing.shape)
    print(df_training['attack_cat'].value_counts()) # 10 classes
    print(df_testing['attack_cat'].value_counts()) # 10 classes
    print("Unique values: ", end="====================\n")
    print(df_training['attack_cat'].unique())
    print(df_testing['attack_cat'].unique())
    print("Number of unique values: ", end="====================\n")
    print(df_training['attack_cat'].nunique())
    print(df_testing['attack_cat'].nunique())

    # import matplotlib.pyplot as plt
    # plot id vs label
    # plot_id_vs_label(df_training, "training")
    # plot_id_vs_label(df_testing, "testing")




