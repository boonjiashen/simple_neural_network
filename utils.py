"""Utility functions to train data"""
import arff  # ARFF module
import itertools

def generate_k_less_one_and_one(items):
    """Generator that yields all items in a list except one, each time

    For example, for a list[1, 2, 3, 4]
    Call #1 yields: [2, 3, 4], 1
    Call #2 yields: [1, 3, 4], 2
    Call #3 yields: [1, 2, 4], 3
    Call #4 yields: [1, 2, 3], 4
    """

    for removed_ind in range(len(items)):
        k_less_one = [x for i, x in enumerate(items) if i != removed_ind]
        one = items[removed_ind]

        yield k_less_one, one


def generate_n_chunks(l, n):
    """Yield n successive chunks from l.
    """

    for i in range(n):
        start_ind = i * len(l) / n
        end_ind = (i + 1) * len(l) / n

        yield l[start_ind: end_ind]


def generate_stratified_k_fold_indices(labels, n_folds):
    """Generates index list pairs for k-fold cross validation.

    Each generation is a list of indices for k-1 folds and a list of indices
    for remaining fold. Labels can only be 0 or 1. Folding is stratified so
    that the label ratio of each fold is approximately preserved.

    For example, if labels are [0, 0, 1, 1, 1, 0] and n_folds==3,
    The folds are [0, 2], [1, 3], [4, 5], and therefore
    Call #1 yields: [1, 3, 4, 5], [0, 2]
    Call #2 yields: [0, 2, 4, 5], [1, 3]
    Call #3 yields: [0, 1, 2, 3], [4, 5]
    """

    # Get n_folds number of folds. Each fold is a list of indices
    folds = [fold for fold in generate_stratified_fold_indices(labels,
        n_folds)]

    assert len(folds) == n_folds

    # In each iteration we lump all but one folds together and yield both
    for i in range(n_folds):
        remaining_fold = folds[i];
        all_but_one_folds = [x
                for fi, fold in enumerate(folds)
                if fi != i
                for x in fold]

        yield all_but_one_folds, remaining_fold


def generate_stratified_fold_indices(labels, n_folds):
    """Generates the indices for one fold, then the second fold, etc

    Labels can only be 0 or 1. Folding is stratified so that the label ratio of
    each fold is approximately preserved.

    For example, if labels are [0, 0, 1, 1, 1, 0] and n_folds==3,
    Call #1 yields: [0, 2]
    Call #2 yields: [1, 3]
    Call #3 yields: [4, 5]
    """

    # Indices of 0 and 1 in labels
    all_indices_for_0 = [i for i, x in enumerate(labels) if x == 0]
    all_indices_for_1 = [i for i, x in enumerate(labels) if x == 1]

    # Yield indices in step-size chunks
    gen_indices_for_0 = generate_n_chunks(all_indices_for_0, n_folds)
    gen_indices_for_1 = generate_n_chunks(all_indices_for_1, n_folds)
    for inds_for_0, inds_for_1 in itertools.izip(gen_indices_for_0,
            gen_indices_for_1):
        yield inds_for_0 + inds_for_1


def load_binary_class_data(arff_file):
    """Returns 2 values - training instances and labels from an ARFF file.

    Labels should have attribute 'Class' in the ARFF file.  We assume that the
    first and second values are the negative and positive labels respectively.
    Then, negative labels are labeled 0 and positive labels are labeled 1.

    Returns:
    X - n_instances-length list of n_feat-length lists
    y - n_instances-length list of integers, either of value 0 or 1.
    """

    # Load ARFF file
    data, metadata = arff.loadarff(arff_file)

    # Get the labels for attribute 'Class'
    # We assume that the first and second values are the negative and positive
    # labels respectively
    norminality, labels = metadata['Class']
    positive_label, negative_label = labels

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
        integer_label = 0 if string_label == negative_label else 1

        # Push into matrices
        X.append(features)
        y.append(integer_label)

    return X, y

