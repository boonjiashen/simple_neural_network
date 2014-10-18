#!/usr/bin/python
"""Implements and trains a simple neural network
"""

import arff  # ARFF module
import optparse
import math

def sigmoid(x):
    "Returns the sigmoid of a number, 1 / (1 + exp(-x))"

    return 1 / (1 + math.exp(-x))

class SimpleNeuralNetwork():
    """Simple neural network classifier.
    
    Architecture is one input layer and one hidden layer (no hidden layer). The
    input layer has a bias unit. The hidden layer consists of one sigmoid unit.

    Cost function is squared error in the output.
    """

    n_feat = 0  # number of features
                # a.k.a dimensionality of one training instance
    W = []  # weights of the connections from the input to hidden layer
            # this is a list of size n_feat
    b = 0  # scalar bias term of the input layer

    def __init__(self, n_feat, initial_weight=0.1):
        """n_feat is the size of the input layer (exluding the bias unit).
        Weights and bias terms are initialized to initial_weight.
        """

        self.n_feat = n_feat
        self.W = [initial_weight for i in range(n_feat)]
        self.b = initial_weight

    def predict(self, X):
        """Predicts the class of X. It predicts 1 if the output of the output
        unit is more than 0.5
        """

        # Logit is the net input to the output unit
        logit = sum([w * x for w, x in zip(self.W, X)])

        predicted_class = 1 if sigmoid(logit) > 0.5 else 0

        return predicted_class

    def calculate_error(self, X, y):
        """Returns the error & error derivative of the weights given a labelled
        instance
        """

        # Calculate the output of the hidden unit from feeding this instance
        logit = sum([w * x for w, x in zip(self.W, X)])
        output = sigmoid(logit)

        # Calculate error of output
        error = 0.5 * (y - output)**2

        # Calculate error derivative of each weight
        dEdw = [-(y - output) * output * (1 - output) * x for x in X]

        return error, dEdw

    def train(self, X, y, learning_rate=0.1):
        """Update weights given one labelled instance (i.e. online training)
        """

        # Get error derivative
        error, dEdws = self.calculate_error(X, y)

        # Get amount that we should update weights by
        dws = [-learning_rate * dEdw for dEdw in dEdws]

        # Update weights
        self.W = [w + dw for w, dw in zip(self.W, dws)]

if __name__ == "__main__":

    # Parse arguments
    parser = optparse.OptionParser()
    options, args = parser.parse_args()
    assert len(args) == 1

    # First positional argument: name of training set file
    filename, = args

    #################### Declare inputs for learning #################### 

    # Load ARFF file
    data, metadata = arff.loadarff(filename)

    # Get the labels for attribute 'Class'
    # We assume that the first and second values are the negative and positive
    # labels respectively
    norminality, labels = metadata['Class']
    positive_label, negative_label = labels

    #################### Pre-process data to be inputs for training ##########

    # One instance loaded from the ARFF file is a list of features followed by
    # the label, which is a string
    n_feat = len(data[0]) - 1

    # Convert ARFF data to X and y
    # X is a list of instances, where each instance is itself a list of
    # features
    # y is a list of labels either 0 or 1
    X, y = [], []
    for instance in data:

        # Split instance into a list of features and a string label
        features = list(instance.tolist()[:-1])
        string_label = instance[-1]

        # Check that string label is one of the possible labels
        assert string_label in (positive_label, negative_label)

        # Convert label from string to 0 or 1
        integer_label = 0 if string_label == negative_label else 0

        # Push into matrices
        X.append(features)
        y.append(integer_label)

    ann = SimpleNeuralNetwork(n_feat)
