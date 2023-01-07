import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.features_list = []
        for column in X.to_numpy().T:
            features_dict = dict()
            uniq_vals = np.unique(column)
            uniq_vals.sort()
            for i, val in enumerate(uniq_vals):
                line = np.zeros(len(uniq_vals))
                line[i] = 1
                features_dict[val] = line
            self.features_list.append(features_dict)

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        res = []
        for row in X.to_numpy():
            res_row = []
            for elem, feature_list in zip(row, self.features_list):
                res_row += list(feature_list[elem])
            res.append(res_row)
        return np.array(res)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.features = dict()
        for key in X:
            self.features[key] = {val: [np.mean(Y.loc[X[key] == val]), len(Y.loc[X[key] == val]) / X.shape[0]]
                                  for val in X[key].unique()}

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        res = np.zeros((X.shape[0], X.shape[1] * 3), dtype=self.dtype)
        for j, key in enumerate(X):
            for i, row in enumerate(X[key]):
                res[i][3 * j], res[i][3 * j + 1], res[i][3 * j + 2] = (self.features[key][row] +
                                                                       [(self.features[key][row][0] + a) /
                                                                        (self.features[key][row][1] + b)])
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.folds = []
        self.features = []
        for i, fold in enumerate(group_k_fold(X.shape[0], self.n_folds, seed)):
            for key in X:
                X_iter, Y_iter = X.iloc[fold[1]], Y.iloc[fold[1]]
                self.features += [dict()]
                self.features[i][key] = {val: [np.mean(Y_iter.loc[X_iter[key] == val]),
                                               len(Y_iter.loc[X_iter[key] == val]) / X_iter.shape[0]]
                                         for val in X_iter[key].unique()}
            self.folds += [fold]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        res = np.zeros((X.shape[0], X.shape[1] * 3), dtype=self.dtype)
        for f, fold in enumerate(self.folds):
            for i in fold[0]:
                for j, key in enumerate(X):
                    row = X.iloc[i][key]
                    res[i][3 * j], res[i][3 * j + 1], res[i][3 * j + 2] = (self.features[f][key][row] +
                                                                           [(self.features[f][key][row][0] + a) /
                                                                            (self.features[f][key][row][1] + b)])
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    encoding = {key: val for val, key in enumerate(np.unique(x))}
    onehot_table = np.zeros((x.shape[0], len(encoding)), dtype=int)
    for i, elem in enumerate(x):
        onehot_table[i][encoding[elem]] = 1
    sum_y1 = np.sum(onehot_table[y == 1], axis=0)
    return sum_y1 / (sum_y1 + np.sum(onehot_table[y == 0], axis=0))
