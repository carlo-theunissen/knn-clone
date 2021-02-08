from collections import Counter
from random import random


def cartesian(x_1, x_2):
    return random()


def knn(train_X, train_y, test_X, metric, k):
    output = []

    for x in test_X:
        distances = [(idx, metric(x, train_X_elem)) for idx, train_X_elem in enumerate(train_X)]
        distances = sorted(distances, key=lambda x: x[1])
        distances = distances[:k]
        nearest_labels = [train_y[distance[0]] for distance in distances]
        most_common = Counter(nearest_labels).most_common()[0][0]  # TODO: break ties systematically

        output.append(most_common)


if __name__ == '__main__':
    pass