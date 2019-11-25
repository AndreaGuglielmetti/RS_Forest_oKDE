# Import libraries
from .ITree import ITree
from concurrent.futures import (ThreadPoolExecutor,
                                wait)
from math import ceil
import numpy as np
from typing import List


# Isolation forest class
class IForest:
    trees: list
    n_estimators: int
    max_samples: int
    max_features: float
    features_weight: np.ndarray
    active_profile: int

    def __init__(self, n_estimators: int = 100, max_samples: int = 256, max_features: float = 1.0,
                 features_weight: np.ndarray = None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.features_weight = features_weight
        self.active_profile = 0
        assert 0. <= self.max_features <= 1.

    # Fit all the required estimators
    def fit(self, X: np.ndarray):
        # Adjust attributes dependent on data
        self.max_samples = min(self.max_samples, X.shape[0])
        self.max_features = ceil(self.max_features * X.shape[1])
        self.features_weight = self.features_weight if self.features_weight else np.ones(X.shape[1]) / X.shape[1]
        assert self.features_weight.ndim == 1 and self.features_weight.shape[0] == X.shape[1] and \
               np.isclose(sum(self.features_weight), 1)
        # Fit each estimator using multithreading
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            futures = []
            for i in range(self.n_estimators):
                samples_indices = np.random.choice(range(X.shape[0]), size=self.max_samples, replace=False)
                features_indices = np.random.choice(range(X.shape[1]), size=int(self.max_features), replace=False)
                futures.append(executor.submit(ITree(features_indices).fit,
                                               X[samples_indices],
                                               self.features_weight[features_indices]/sum(self.features_weight[features_indices])))
            wait(futures)
            self.trees = [future.result() for future in futures]

    # Profile samples
    def profile(self, X: np.ndarray):
        # Test all samples through each estimator using multithreading
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            futures = []
            for i in range(self.n_estimators):
                futures.append(executor.submit(self.trees[i].profile, X, self.active_profile))
            wait(futures)
            depths = np.asarray([future.result() for future in futures]).T
        return depths

    def score(self, X: np.ndarray):
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            futures = []
            for i in range(self.n_estimators):
                futures.append(executor.submit(self.trees[i].score, X, self.active_profile))
            wait(futures)
            score = np.zeros(X.shape[0])
            arrival_node_per_tree = []
            for future in futures:
                leaf_size, log_scaled_ratio, arrival_node = future.result()
                score += self._compute_score(leaf_size, log_scaled_ratio)
                arrival_node_per_tree.append(arrival_node)
        return score, arrival_node_per_tree

    def _compute_score(self, leaf_size: np.ndarray, log_scaled_ratio: np.ndarray):
        '''
        Computes the density associated to each samples based on the node in which they fall.
        :param leaf_size: ndarray (n_samples, ), the number of samples contained in the leaf where sample falls.
        :param log_scale_ratio: ndarray (n_samples, ), the leaf log-scaled volume ratio where sample falls.
        :return:
        '''
        return np.exp(np.log(leaf_size) - log_scaled_ratio - np.log(self.max_samples))

    def update_model(self, arrival_nodes_per_tree: List[List['INode']], is_anomaly: List[bool], X: np.ndarray):
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            futures = []
            for i in range(self.n_estimators):
                arrival_nodes = arrival_nodes_per_tree[i]
                futures.append(executor.submit(self.trees[i].update_tree, arrival_nodes, is_anomaly, X, self.active_profile))
            wait(futures)
            self.active_profile = abs(self.active_profile - 1) # change from 0 to 1 and viceversa
