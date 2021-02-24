import multiprocessing
import os
import zipfile

import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import time
from tabulate import tabulate
from sklearn.decomposition import TruncatedSVD
from KNN import k_nn


def knn(train_X, train_y, test_X, distance, p=0):
    return k_nn(train_X, train_y, test_X, distance, p)


def run_parallel_knn(train_data_x, train_data_y, validation_x, distance, p=0):
    n_cores = multiprocessing.cpu_count()
    return Parallel(n_jobs=n_cores)(delayed(knn)(train_data_x, train_data_y, test_point, distance, p=0)
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


def exercise_b():
    correct = [0 for _ in range(20)]

    # combine the test and training set
    joined_train_x = np.concatenate((train_data_x, test_data_x), axis=0)
    joined_train_y = np.concatenate((train_data_y, test_data_y), axis=0)

    # now, for every item of this combined set, verify if the k-means algorithm predicts a good label
    for i in range(len(joined_train_x)):

        # save the to-be removed value in dummy variables
        dummy_x = joined_train_x[i]
        dummy_y = joined_train_y[i]

        # remove the dummy variables
        train_x = np.delete(joined_train_x, i, 0)
        train_y = np.delete(joined_train_y, i)

        # runs the the knn algorithm
        predicted = knn(train_x, train_y, dummy_x, 'euclidean')

        # count for every k setting if the predicted label is equal to the real one
        for k in range(1, 21):
            correct[k-1] += predicted.get_prediction(k) == dummy_y

    max_correct_value = max(correct)
    print(tabulate(
        [[k, correct[k-1] / joined_train_x.shape[0], int(correct[k-1] == max_correct_value)] for k in range(1,21)],
        headers=['K', 'Accuracy', 'Is Highest']))


def exercise_b_without_test_set():
    correct = [0 for _ in range(20)]

    # now, for every item of the training set, verify if the k-means algorithm predicts a good label
    for i in range(len(train_data_x)):

        # save the to-be removed value in dummy variables
        dummy_x = train_data_x[i]
        dummy_y = train_data_y[i]

        # remove the dummy variables
        train_x = np.delete(train_data_x, i, 0)
        train_y = np.delete(train_data_y, i)

        # runs the the knn algorithm
        predicted = knn(train_x, train_y, dummy_x, 'euclidean')

        # count for every k setting if the predicted label is equal to the real one
        for k in range(1, 21):
            correct[k-1] += predicted.get_prediction(k) == dummy_y

    max_correct_value = max(correct)
    print(tabulate(
        [[k, correct[k-1] / train_data_x.shape[0], int(correct[k-1] == max_correct_value)] for k in range(1,21)],
        headers=['K', 'Accuracy', 'Is Highest']))
    

def exercise_c():
    optimal_p, optimal_k, highest_accuracy = 0, 0, 0
    for p in range(15):
        correct = [0 for _ in range(20)]

        # now, for every item of the training set, verify if the k-means algorithm predicts a good label
        for i in range(len(train_data_x)):

            # save the to-be removed value in dummy variables
            dummy_x = train_data_x[i]
            dummy_y = train_data_y[i]

            # remove the dummy variables
            train_x = np.delete(train_data_x, i, 0)
            train_y = np.delete(train_data_y, i)

            # runs the the knn algorithm
            predicted = knn(train_x, train_y, dummy_x, 'minkowski', p+1)

            # count for every k setting if the predicted label is equal to the real one
            for k in range(1, 21):
                correct[k-1] += predicted.get_prediction(k) == dummy_y
                
                max_correct_value = max(correct)
                
                current_accuracy = correct[k-1] / train_data_x.shape[0]
                
                if(current_accuracy > highest_accuracy):
                    optimal_p = p + 1
                    optimal_k = k
                    highest_accuracy = current_accuracy
        
        print(('------p={p_value}------').format(p_value=p+1))
        print(tabulate(
            [[k, correct[k-1] / train_data_x.shape[0], int(correct[k-1] == max_correct_value)] for k in range(1,21)],
            headers=['K', 'Accuracy', 'Is Highest']))
        print('------------------------\n')
    
    print('OPTIMAL VALUES FOR P AND K: {opt_p} and {opt_k} with an accuracy of {accu}'.format(opt_p=optimal_p, opt_k=optimal_k, accu=highest_accuracy))


def exercise_e():
    n_cores = multiprocessing.cpu_count()

    # now, for every item of the training set, verify if the k-means algorithm predicts a good label
    def job(i):
        correct = [0 for _ in range(20)]
        print(i)

        # save the to-be removed value in dummy variables
        dummy_x = train_data_x_large[i]
        dummy_y = train_data_y_large[i]

        # remove the dummy variables
        train_x = np.delete(train_data_x_large, i, 0)
        train_y = np.delete(train_data_y_large, i)


        # runs the the knn algorithm
        predicted = knn(train_x, train_y, dummy_x, 'euclidean')

        # count for every k setting if the predicted label is equal to the real one
        for k in range(1, 21):
            correct[k-1] = predicted.get_prediction(k) == dummy_y

        return correct

    list_of_corrects = Parallel(n_jobs=n_cores)(delayed(job)(i) for i in range(len(train_data_x_large)))
    correct = np.array(list_of_corrects).sum(axis=0)

    max_correct_value = max(correct)
    print(tabulate(
        [[k, correct[k - 1] / train_data_x_large.shape[0], int(correct[k - 1] == max_correct_value)] for k in range(1, 21)],
        headers=['K', 'Accuracy', 'Is Highest']))


def exercise_g(n_components=18):
    global train_data_x
    global test_data_x

    svd = TruncatedSVD(n_components=n_components)
    svd.fit(train_data_x)
    train_data_x = svd.transform(train_data_x)
    test_data_x = svd.transform(test_data_x)

        
if __name__ == '__main__':
    # Unpack the archive if necessary
    if not os.path.exists("MNIST_train.csv") and not os.path.exists("MNIST_test.csv"):
        with zipfile.ZipFile("MNIST_big_datasets.zip", 'r') as zip_ref:
            zip_ref.extractall()

    train_data_x = pd.read_csv("MNIST_train_small.csv").to_numpy()[:, 1:]
    train_data_y = pd.read_csv("MNIST_train_small.csv").to_numpy()[:, 0]
    test_data_x = pd.read_csv("MNIST_test_small.csv").to_numpy()[:, 1:]
    test_data_y = pd.read_csv("MNIST_test_small.csv").to_numpy()[:, 0]

    train_data_x_large = pd.read_csv("MNIST_train.csv").to_numpy()[:60000, 1:]
    train_data_y_large = pd.read_csv("MNIST_train.csv").to_numpy()[:60000, 0]
    test_data_x_large = pd.read_csv("MNIST_train.csv").to_numpy()[:60000, 1:]
    test_data_y_large = pd.read_csv("MNIST_train.csv").to_numpy()[:60000, 0]

    t0 = time.time()

    exercise_e()
    print(time.time() - t0)
