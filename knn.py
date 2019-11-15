import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split


def dummy():
    print("hellooo world!")

# calculate the euclidean distance


def euclidean_distance(x1, x2):
    np.sqrt(np.sum((x1-x2)**2))

# to read training and testing datasets


def read_file(file):
    df = pd.read_csv(file)
    return df

# to write the output to a file


def output_file(file):
    file = open(file, 'w')
    # to be finished later


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def fit(self, X, y):
    self.X_train = X
    self.y_train = y


def predict(self, X):
    predicted_labels = [self._predcit(x) for x in X]
    return np.array(predicted_labels)


def _predcit(self, x):
        # distances
    distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
    # get the nearest neighbor
    k_index = np.argsort(distance)[:self.k]
    k_nearest_label = [self.y_train[i] for i in k_index]

    # get the most common class label
    most_common = Counter(k_nearest_label).most_common(1)
    return most_common[0][0]


if __name__ == "__main__":
    training_data = read_file("buyTraining.txt")
    testing_data = read_file("buytesting.txt")
    print(training_data, testing_data)
