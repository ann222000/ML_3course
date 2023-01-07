import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    fold_size = num_objects // num_folds
    return [(np.array([j for j in range(num_objects) if j not in range(i * fold_size, (i+1) * fold_size)]),
             np.array([j for j in range(i * fold_size, (i + 1) * fold_size)])) for i in range(num_folds - 1)] + \
           [(np.array([j for j in range((num_folds - 1) * fold_size)]), np.array([j for j in range((num_folds - 1) * fold_size, num_objects)]))]


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    result = dict()
    for n_neighbors in parameters['n_neighbors']:
        for metrics in parameters['metrics']:
            for weight in parameters['weights']:
                for normalizer in parameters['normalizers']:
                    knn_obj = knn_class(n_neighbors=n_neighbors, weights=weight, metric=metrics)
                    score = []
                    for folds_iter in folds:
                        if normalizer[0]:
                            normalizer[0].fit(X[folds_iter[0]])
                            normalized_test_data = normalizer[0].transform(X[folds_iter[1]])
                            normalized_train_data = normalizer[0].transform(X[folds_iter[0]])
                        else:
                            normalized_test_data = X[folds_iter[1]]
                            normalized_train_data = X[folds_iter[0]]
                        knn_obj.fit(normalized_train_data, y[folds_iter[0]])
                        y_predict = knn_obj.predict(normalized_test_data)
                        score.append(score_function(y[folds_iter[1]], y_predict))
                    result[(normalizer[1], n_neighbors, metrics, weight)] = np.mean(score)
    return result
