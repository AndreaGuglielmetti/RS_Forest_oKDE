# Import libraries
from .INode import INode
from math import ceil
import numpy as np
from typing import List


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
        # self.root = INode().fit(X, 0, ceil(np.log2(X.shape[0])), features_weight)
        self.root = INode().fit(X, 0, 4, features_weight)
        return self

    # Profile all the passed samples
    def profile(self, X: np.ndarray, active_profile: int):
        # Keep only features selected
        X = X[:, self.features_indices]
        # Test all samples, one at a time
        depths = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            depths[i] = self.root.profile(x, 0, active_profile)
        return depths

    def score(self, X: np.ndarray, active_profile: int):
        '''
        :param X: ndarray, shape(n_samples, n_dimensions)
        :return: tuple(ndarray, ndarray), the first ndarray is the leaf cardinality which contains the sample x,
            the second is the log-scaled ratio of the leaf volume with the entire volume
        '''
        X = X[:, self.features_indices]
        leaf_size = np.empty((X.shape[0]))
        log_scaled_ratio = np.empty((X.shape[0]))
        for i, x in enumerate(X):
            leaf_size[i], log_scaled_ratio[i] = self.root.score(x, active_profile)
        return leaf_size, log_scaled_ratio

    def get_terminal_node(self, X: np.ndarray, active_profile: int):
        terminal_nodes = [self.root.get_terminal_node(x, active_profile) for x in X]

    def update_tree(self, arrival_nodes: List[INode], is_anomaly: List[bool], X: np.ndarray, active_profile: int):
        for i, arrival_node in enumerate(arrival_nodes):
            arrival_node.update_path(is_anomaly[i], X[i], active_profile)

    def reset_profile(self, active_profile: int):
        self.root.reset_profile(active_profile)
