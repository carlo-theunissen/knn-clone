import multiprocessing
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import time
from tabulate import tabulate

from KMeans import k_means


def knn(train_X, train_y, test_X):
    return k_means(train_X, train_y, test_X)


def run_parallel_knn(train_data_x, train_data_y, validation_x):
    n_cores = multiprocessing.cpu_count()
    return Parallel(n_jobs=n_cores)(delayed(knn)(train_data_x, train_data_y, test_point)
                                    for _, test_point in enumerate(validation_x))


def get_accuracy(predicted, actual):
    """Returns the accuracy of the predicted values, compared to the actual ones.
    The accuracy is in the range [0,1]. Where as 0 indicates "no values of the predicted list are equal to those of
    actual", and 1 indicates "all values are equal".
    """

    diff = 0
    for i in range(len(predicted)):
        diff += (predicted[i] != actual[i])

    return 1 - diff / len(predicted)


def exercise_a():
    """
    Fetches the small training set and uses the K-mean algorithm to predict
    the test data set. The k parameter is increased from 1 to 20. Finally
    a table with the results are printed.
    :return: none
    """
    result = []
    highest_index = -1
    predicted = run_parallel_knn(train_data_x, train_data_y, test_data_x)

    # run through all ks from 1 to 20, compare the output to calculate the accuracy
    # and put it in the result list.
    for k in range(1, 21):
        accuracy = get_accuracy(list(map(lambda x: x.get_prediction(k), predicted)), test_data_y)

        if highest_index == -1 or result[highest_index][1] < accuracy:
            highest_index = k - 1

        result.append(
            # k | accuracy | dummy value
            [k, accuracy, 0])

    # print a tabulate with the fetched information
    result[highest_index][2] = 1  # set the dummy value to 1, indicating this index is the highest
    print(tabulate(result, headers=['K', 'Accuracy', 'Is Highest']))


if __name__ == '__main__':
    train_data_x = np.repeat(pd.read_csv("MNIST_train_small.csv").to_numpy()[:, 1:], repeats=1, axis=0)
    train_data_y = np.repeat(pd.read_csv("MNIST_train_small.csv").to_numpy()[:, 0], repeats=1, axis=0)
    test_data_x = np.repeat(pd.read_csv("MNIST_test_small.csv").to_numpy()[:, 1:], repeats=1, axis=0)
    test_data_y = np.repeat(pd.read_csv("MNIST_test_small.csv").to_numpy()[:, 0], repeats=1, axis=0)
    t0 = time.time()

    exercise_a()
    print(time.time() - t0)
