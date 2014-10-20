"""Plot ROC curve

Plot an ROC curve for a run of 10-fold cross validation with a learning rate of
0.1 and 100 training epochs. You should pool the classifications from the 10
test sets to make one curve. Use the activation of the output unit (i.e. the
value computed by the sigmoid) as the measure of confidence that a given test
instance is positive. You should consider 'Mine' to be the positive class.
"""

import os
import matplotlib.pyplot as plt

n_folds = 10 
learning_rate = 0.1
n_epochs = 100

tmp_filename = 'tmp'

def write_data_file():

    ##################### Generate testset results ####################

    stdout_cmd = "./neuralnet data/sonar.arff %i %f %i" % (n_folds, learning_rate,
            n_epochs)
    os.system('rm ' + tmp_filename)
    os.system(stdout_cmd + " >> " + tmp_filename)


if __name__ == "__main__":

    # Get true labels and confidence in prediction of a dataset
    # -------------------- 
    # Our program should do stratified cross validation with the specified
    # number of folds. As output, it should print one line for each instance
    # (in the same order as the data file) indicating (i) the fold to which the
    # instance was assigned (1 to n), (ii) the predicted class, (iii) the
    # actual class, (iv) the confidence of the instance being positive (i.e.
    # the output of the sigmoid).
    truths, confidences = [], []
    with open(tmp_filename, 'r') as fid:

        for line in fid:
            true_label = line.split()[-2] == 'Mine'
            confidence = float(line.split()[-1])

            truths.append(true_label)
            confidences.append(confidence)

    # Sort dataset by descending order of confidence
    truth_conf_tuples = zip(truths, confidences)
    truth_conf_tuples.sort(key=lambda x: x[1], reverse=True)
    truths, confidences = zip(*truth_conf_tuples)


    #################### Generate data points of ROC  ####################    

    ROC_coordinates = []  # list of FPR, TPR tuples
    n_FP, n_TP = 0, 0  # number of false and truth positives
    n_instances = len(truths)
    n_neg, n_pos = truths.count(False), truths.count(True)
    for ii in range(n_instances):

        truth, confidence = truths[ii], confidences[ii]

        # Decide if we generate a data point here
        # Either it's the first instance (i.e. most confidence instance) or
        # this is a negative instance and the previous instance was positive
        is_first = ii == 0
        curr_neg_prev_pos = not is_first and not truth and truths[ii - 1]
        if is_first or curr_neg_prev_pos:
            FPR = 1. * n_FP / n_neg
            TPR = 1. * n_TP / n_pos

            ROC_coordinates.append((FPR, TPR, ))

        # Update no. of TP and FP
        if truth:
            n_TP = n_TP + 1
        else:
            n_FP = n_FP + 1


    #################### Plot ROC ####################    

    # Plot curve
    x, y = zip(*ROC_coordinates)
    plt.plot(x, y)

    # Prettify plot
    title_line1 = "ROC curve of a simple trained neural network"
    title_line2 = "%i folds, %i epochs, learning rate is %f" %  \
            (n_folds, n_epochs, learning_rate)
    plt.title(title_line1 + "\n" + title_line2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.axis([0, 1, 0, 1])

    plt.show()
