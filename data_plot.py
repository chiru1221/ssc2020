import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def read_data(name):
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sub_df = pd.read_csv('submit_sample.csv', header=None)
    submit_df = pd.read_csv('submit/' + name + '.csv', header=None)
    return train_df, test_df, sub_df, submit_df

def main():
    train_df, test_df, sub_df, submit_df = read_data('bnn')
    plt.figure()
    plt.hist(train_df.iloc[:, 2].values)
    plt.savefig('train_label.jpg')
    plt.figure()
    plt.hist(submit_df.iloc[:, 1].values)
    plt.savefig('bnn.jpg')

if __name__ == '__main__':
    main()