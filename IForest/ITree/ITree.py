# Import libraries
from .INode import INode
from math import ceil
import numpy as np


# ITree class
class ITree:
    features_indices: np.ndarray
    root: INode

    def __init__(self, features_indices: np.ndarray):
        self.features_indices = features_indices

    # Build the tree
    def fit(self, X: np.ndarray, features_weight: np.ndarray):
        # Keep only features selected
        X = X[:, self.features_indices]
        # Build all nodes of the tree
        self.root = INode().fit(X, 0, ceil(np.log2(X.shape[0])), features_weight)
        return self

    # Profile all the passed samples
    def profile(self, X: np.ndarray):
        # Keep only features selected
        X = X[:, self.features_indices]
        # Test all samples, one at a time
        depths = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            depths[i] = self.root.profile(x, 0)
        return depths

    def score(self, X: np.ndarray):
        '''
        :param X: ndarray, shape(n_samples, n_dimensions)
        :return: ndarray, shape(n_samples, n_dimensions), the first column is the leaf cardinality which contains the
        sample x, the second column is the log ratio of the leaf volume with features volume
        '''
        X = X[:, self.features_indices]
        scores = np.zeros((X.shape[0], self.feature_indices.shape))
        for i, x in enumerate(X):
            scores[i] = self.root.score(x)
        return scores
