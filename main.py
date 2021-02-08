from collections import Counter
from random import random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def cartesian(x_1, x_2):
    diffs = sum([(comp_x_1i - comp_x_2i)**2 for comp_x_1i, comp_x_2i in zip(x_1, x_2)])
    return math.sqrt(diffs)


def knn(train_X, train_y, test_X, metric, k):
    output = []
    i = 1

    for x in test_X:
        distances = [(idx, metric(x, train_X_elem)) for idx, train_X_elem in enumerate(train_X)]
        distances = sorted(distances, key=lambda x: x[1])
        distances = distances[:k]
        nearest_labels = [train_y[distance[0]] for distance in distances]
        most_common = Counter(nearest_labels).most_common()[0][0]  # TODO: break ties systematically

        output.append(most_common)

        print(i)
        i += 1

    return output


if __name__ == '__main__':
    train_data_X = pd.read_csv("MNIST_train_small.csv").to_numpy()[:100, 1:2]
    train_data_y = pd.read_csv("MNIST_train_small.csv").to_numpy()[:100, 0]
    test_data_X = pd.read_csv("MNIST_test_small.csv").to_numpy()[:100, 1:]
    test_data_y = pd.read_csv("MNIST_test_small.csv").to_numpy()[:100, 0]

    results = knn(train_data_X, train_data_y, test_data_X, cartesian, k=3)

    plt.tight_layout()

    fig, axs = plt.subplots(10, 4, figsize=(50, 100))
    axs = axs.ravel()

    # Pick items at random to show
    for i in range(10*4):
        idx = np.random.randint(len(results))
        result = results[idx]

        axs[i].imshow(test_data_X[idx].reshape(28, 28))
        axs[i].set_label(f"predicted: {result}, actual: {test_data_y[idx]}")

    plt.show()
