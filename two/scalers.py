import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.min = np.min(data, axis=0)
        self.max_minus_min = np.max(data, axis=0) - self.min

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        self.max_minus_min[np.where(self.max_minus_min == 0)] = 1
        return (data - self.min) / self.max_minus_min


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.math_expection = np.sum(data, axis=0) / len(data)
        self.sigma = (np.sum(data * data, axis=0) / len(data) - self.math_expection ** 2) ** 0.5

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        self.sigma[np.where(self.sigma == 0)] = 1
        return (data - self.math_expection) / self.sigma
