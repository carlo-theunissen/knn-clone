import math
from collections import Counter


def cartesian(x_1, x_2):
    squared_diffs = (x_1 - x_2) ** 2
    return math.sqrt(squared_diffs.sum())


def k_means(train_data_x, train_data_y, test_element_x):
    distances = [(idx, cartesian(test_element_x, train_element))
                 for idx, train_element in enumerate(train_data_x)]

    distances = sorted(distances, key=lambda x: x[1])
    return KMeansResult(distances=distances, train_data_y=train_data_y)


class KMeansResult:
    distances = []

    def __init__(self, distances,train_data_y):
        self.distances = distances
        self.train_data_y = train_data_y

    def get_prediction(self, k):
        copy = self.distances[:k]
        # Get the labels of the nearest neighbours
        nearest_labels = [self.train_data_y[distance[0]] for distance in copy]
        # Break ties in ascending order
        most_commons = Counter(nearest_labels).most_common()
        return most_commons[0][0]
