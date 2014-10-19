"""Plot learning curves for a simple neural network
"""

import os
import matplotlib.pyplot as plt

n_folds = 10 
learning_rate = 0.1
list_of_n_epochs = [1, 10, 100, 1000]
filename = "train_and_test_accuracies"


def generate_data_points():
    """Generate data points for learning curve

    Each line in the file is the training accuracy followed by the test
    accuracy"""

    os.system('rm ' + filename)

    # Generate plot points
    for n_epochs in list_of_n_epochs:

        # Generate test results for a particular n_epochs
        tmp_filename = 'tmp'
        stdout_cmd = "./neuralnet data/sonar.arff %i %f %i" % (n_folds, learning_rate,
                n_epochs)
        os.system('rm ' + tmp_filename)
        os.system(stdout_cmd + " >> " + tmp_filename)

        # Calculate test_accuracy for this particular n_epochs
        with open(tmp_filename, 'r') as fid:
            n_predictions = 0
            n_correct_predictions = 0

            for line in fid:
                prediction, true_label = line.split()[1:3]
                n_predictions = n_predictions + 1
                n_correct_predictions = n_correct_predictions +  \
                        (prediction == true_label)

            test_accuracy = n_correct_predictions * 1. / n_predictions

        # Write test accuracy to file
        with open(filename, 'a') as fid:
            fid.write('%f ' % test_accuracy)


def plot():
    """Plot learning curves"""

    # Get data points
    with open(filename, 'r') as fid:
        test_accuracies = map(float, fid.next().split())

    print test_accuracies

    # Plot data points
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
