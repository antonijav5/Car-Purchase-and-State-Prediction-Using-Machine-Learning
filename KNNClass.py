from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sb
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, BallTree, KDTree


class KNNClass:

    def __init__(self, n_neighbors=5, algorithm="brute", leaf_size=30):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size

    def fit(self, X, y):
        self._fit_X = X
        self.classes_, self._fit_y = scipy.unique(y, return_inverse=True)

        if self.algorithm == "brute":
            pass
        elif self.algorithm == "kd_tree":
            self.tree = KDTree(X, leaf_size=self.leaf_size)
        elif self.algorithm == "ball_tree":
            self.tree = BallTree(X, leaf_size=self.leaf_size)
        else:
            raise ValueError("unrecognized algorithm: ", str(self.algorithm))
        return self

    def predict(self, X_train, y_train, x_test, k):

        distances = []
        targets = []

        for i in range(len(X_train)):

            distances.append([np.sqrt(np.sum(np.square(x_test - X_train[i, :]))), i])


        distances = sorted(distances)

        for i in range(k):
            index = distances[i][1]
            targets.append(y_train[index])


        return Counter(targets).most_common(1)[0][0]

    def train(self, X_train, y_train):
        # do nothing
        return

    def k_nearest_neighbor(self, X_train, y_train, X_test, k):

        self.train(X_train, y_train)

        predictions = []
        for i in range(len(X_test)):
            predictions.append(self.predict(X_train, y_train, X_test[i, :], k))

        return np.asarray(predictions)