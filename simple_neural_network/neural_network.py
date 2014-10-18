#!/usr/bin/python
"""Perform cross-validation to get test and training accuracy on a neural
network.
"""
import optparse
import math
import random
import utils
from SimpleNeuralNetwork import SimpleNeuralNetwork


if __name__ == "__main__":


    ###################### Parse arguments ######################

    parser = optparse.OptionParser()
    options, args = parser.parse_args()
    assert len(args) == 4

    # Positional argument 1: name of training set file
    # Positional argument 2: number of folds for cross-validation
    # Positional argument 3: learning rate of stochastic gradient descent
    # Positional argument 4: number of epochs for training
    filename, n_folds, learning_rate, n_epochs = args
    n_folds = int(n_folds)
    learning_rate = float(learning_rate)
    n_epochs = int(n_epochs)


    #################### Declare inputs for learning #################### 

    # Load dataset X and labels y from ARFF file
    X, y = utils.load_binary_class_data(filename)

    # Size of data set
    n_instances = len(X)

    # Size of each feature vector
    n_feat = len(X[0])


    #################### Train neural network#################### 

    # Shuffle dataset
    indices = range(n_instances)
    random.shuffle(indices)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    def get_L2_norm(vector):
        return sum([x**2 for x in vector])**.5

    # Perform k folds cross validation
    train_accuracies, test_accuracies = [], []
    for training_inds, test_inds in  \
            utils.generate_stratified_k_fold_indices(y, n_folds):

        # Initialize network
        ann = SimpleNeuralNetwork(n_feat)

        # Train network by stochastic gradient descent
        for training_ind in n_epochs * training_inds:
            instance, label = X[training_ind], y[training_ind]
            ann.train(instance, label, learning_rate=learning_rate)

        # Calculate training accuracy
        X_train = [X[i] for i in training_inds]
        y_train = [y[i] for i in training_inds]
        train_accuracy = ann.calculate_accuracy(X_train, y_train)

        # Calculate test accuracy
        X_test = [X[i] for i in test_inds]
        y_test = [y[i] for i in test_inds]
        test_accuracy = ann.calculate_accuracy(X_test, y_test)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    ave_train_accuracy = 1. * sum(train_accuracies) / len(train_accuracies)
    ave_test_accuracy = 1. * sum(test_accuracies) / len(test_accuracies)

    print ave_train_accuracy, ave_test_accuracy
