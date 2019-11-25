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
        :return: tuple(ndarray, ndarray, INode)
        '''
        X = X[:, self.features_indices]
        leaf_size = np.empty(X.shape[0])
        log_scaled_ratio = np.empty(X.shape[0])
        arrival_nodes = []
        for i, x in enumerate(X):
            leaf_size[i], log_scaled_ratio[i], node = self.root.score(x)
            arrival_nodes.append(node)
        return leaf_size, log_scaled_ratio, arrival_nodes
