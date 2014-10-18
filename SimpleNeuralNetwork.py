"""Implements a simple neural network
"""
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

    def calculate_accuracy(self, X, y):
        """Calculate the accuracy of the model on a labelled dataset
        """

        predictions = [self.predict(x) for x in X]
        n_correct_predictions = sum([prediction == truth
            for prediction, truth in zip(predictions, y)
            ])
        accuracy = 1. * n_correct_predictions / len(X)

        return accuracy
