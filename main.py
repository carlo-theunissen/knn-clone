import multiprocessing
from collections import Counter
import math
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def cartesian(x_1, x_2):
    squared_diffs = (x_1 - x_2)**2
    return math.sqrt(squared_diffs.sum())


def knn(train_X, train_y, test_X, metric, k, current_idx=None):
    if current_idx is not None:
        print(current_idx)

    output = []

    for x in test_X:
        # Compute the distances
        distances = [(idx, metric(x, train_X_elem)) for idx, train_X_elem in enumerate(train_X)]
        distances = sorted(distances, key=lambda x: x[1])
        distances = distances[:k]

        # Get the labels of the nearest neighbours
        nearest_labels = [train_y[distance[0]] for distance in distances]
        # Break ties in ascending order
        most_commons = sorted(Counter(nearest_labels).most_common(), key=lambda x: (-x[1], x[0]))
        most_common = most_commons[0][0]

        output.append(most_common)

    return output


if __name__ == '__main__':
    train_data_X = pd.read_csv("MNIST_train_small.csv").to_numpy()[:, 1:]
    train_data_y = pd.read_csv("MNIST_train_small.csv").to_numpy()[:, 0]
    test_data_X = pd.read_csv("MNIST_test_small.csv").to_numpy()[:, 1:]
    test_data_y = pd.read_csv("MNIST_test_small.csv").to_numpy()[:, 0]

    n_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=n_cores)(delayed(knn)(train_data_X, train_data_y, [test_point], cartesian, 3, idx)
                                       for idx, test_point in enumerate(test_data_X))
