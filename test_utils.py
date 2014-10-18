"""Test utility functions
"""

import utils
import random

random.seed(0)

def get_0_to_1_ratio(labels):
    return 1.0 * labels.count(0) / labels.count(1)


def testStratifiedTrainTestPreserveLabelRatio():

    # Create labels of some label ratio that's easily divided
    labels = 6 * [0] + 3 * [1]
    random.shuffle(labels)

    # Generate partition schemes
    # Each scheme is a tuple of training indices and test indices
    generator = utils.generate_stratified_train_test_indices(labels, 3,
            randomize=True)
    partition_schemes = [scheme for scheme in generator]

    # Check that 0 to 1 ratio is same for all training and test indices
    ratios = [get_0_to_1_ratio([labels[i] for i in indices])
            for scheme in partition_schemes
            for indices in scheme]
    assert ratios.count(ratios[0]) == len(ratios)


def testStratifiedFoldsPreserveLabelRatio():

    # Create labels of some label ratio that's easily divided
    labels = 6 * [0] + 3 * [1]
    random.shuffle(labels)

    # Generate folds
    generator = utils.generate_stratified_fold_indices(labels, 3,
            randomize=True)
    folds = [fold for fold in generator]

    # Check that 0 to 1 ratio is same for all folds
    ratios = [get_0_to_1_ratio([labels[i] for i in fold])
            for fold in folds]
    assert ratios.count(ratios[0]) == len(ratios)


if __name__ == "__main__":

    # Get all functions that start with test
    test_keys = [key for key in globals().keys() if key[:4] == 'test']

    # Run all such function
    for key in test_keys:
        globals()[key]()
