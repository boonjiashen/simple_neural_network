"""Plot learning curves for a simple neural network
"""

import os
import matplotlib.pyplot as plt

n_folds = 3
learning_rate = 0.1
list_of_n_epochs = [1, 10, 100, 1000]
filename = "train_and_test_accuracies"


def generate_data_points():
    """Generate data points for learning curve

    Each line in the file is the training accuracy followed by the test
    accuracy"""

    # Generate plot points
    for n_epochs in list_of_n_epochs:

        stdout_cmd = "./neuralnet data/sonar.arff %i %f %i" % (n_folds, learning_rate,
                n_epochs)
        os.system(stdout_cmd + " >> " + filename)


def plot():
    """Plot learning curves"""

    # Load data points from file
    with open(filename, 'r') as fid:
        train_and_test_pairs = [map(float, line.split()) for line in fid]
    train_accuracies, test_accuracies = zip(*train_and_test_pairs)

    # Plot data points
    plt.plot(list_of_n_epochs, train_accuracies, label="Training accuracy")
    plt.plot(list_of_n_epochs, test_accuracies, label="Test accuracy")

    # Prettify plot
    plt.legend(loc='best')
    plt.semilogx(10)
    plt.ylim([0, 1])
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of a simple neural network\n' +  \
            'versus number of training epochs')

    # Change figure title
    fig = plt.gcf()
    fig_title = "no. of folds = %i, learning rate is %f" %  \
            (n_folds, learning_rate)
    fig.canvas.set_window_title(fig_title)

    # Show figure
    plt.show()
