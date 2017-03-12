""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    num_trials = 10
    train_percentages = range(5, 95, 5)
    print(train_percentages)
    test_accuracies = []
    for i in train_percentages:
        test_accuracy = 0
        total = 0
        avg = 0
        for j in range(0, 20):
            data = load_digits()
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size= i/100)
            model = LogisticRegression(C=10**-15)
            model.fit(X_train, y_train)
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            total = test_accuracy + total
            avg += 1
        clean = total/avg
        print(clean)
        plotdata = numpy.append(test_accuracies, clean)
        test_accuracies = plotdata


    fig = plt.figure()
    plt.plot(train_percentages, plotdata)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #display_digits()
    train_model()
