from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from .RSTree import RSTreeArrayBased
from .RSTree.RSNode import RSNode
from typing import List
from math import log, ceil


class RSForest:
    trees: List[RSTreeArrayBased]
    n_estimators: int
    max_samples: int
    log_max_samples: float
    current_profile: int
    feature_volume: float
    max_node_size: int

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, max_samples: int = 256, max_node_size: float = 1.):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.current_profile = 0
        self.feature_volume = 0.
        self.max_node_size = ceil(max_samples * max_node_size)

    def fit(self, samples: np.ndarray, enlarge_bounds: bool = False):
        self.max_samples = min(self.max_samples, samples.shape[0])
        self.log_max_samples = log(self.max_samples)
        if samples.shape[0] > self.max_samples:
            indices = np.random.choice(range(samples.shape[0]), size=self.max_samples, replace=False)
        else:
            indices = np.arange(self.max_samples)
        bounds = self._compute_bounds(samples, enlarge_bounds)
        self.feature_volume = float(np.prod(np.diff(bounds, axis=1)))

        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            futures = []
            for i in range(self.n_estimators):
                futures.append(executor.submit(RSTreeArrayBased(self.max_depth, self.max_node_size).fit,
                                               bounds, samples[indices]))
            wait(futures)
        self.trees = [future.result() for future in futures]

    @staticmethod
    def _compute_bounds(samples: np.ndarray, enlarge: bool):
        if not enlarge:
            return np.apply_along_axis(lambda x: [x.min(), x.max()], axis=0, arr=samples).T
        else:
            means = np.mean(samples, axis=0)
            std_deviations = np.std(samples, axis=0)
            enlarged_lbound = means - 4.645 * std_deviations
            enlarged_ubound = means + 4.645 * std_deviations
            return np.hstack((
                enlarged_lbound.reshape((-1, 1)),
                enlarged_ubound.reshape((-1, 1))
            ))

    def score(self, samples: np.ndarray, normalize=False):
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            for i in range(self.n_estimators):
                futures.append(executor.submit(self.trees[i].score, samples, self.current_profile))
        wait(futures)

        score = np.zeros(samples.shape[0])
        for future in futures:
            leaf_size, log_scaled_ratio = future.result()
            score += self._compute_score(leaf_size, log_scaled_ratio)

        if normalize:
            score /= (self.n_estimators * self.feature_volume)

        return score

    def _compute_score(self, leaf_size, log_scaled_ratio):
        not_zero_leaves = np.nonzero(leaf_size)[0]
        scores = np.zeros(leaf_size.shape[0])
        scores[not_zero_leaves] = np.exp(
            np.log(leaf_size[not_zero_leaves]) - log_scaled_ratio[not_zero_leaves] - self.log_max_samples)
        return scores

    def update_forest(self, samples: np.ndarray, arrival_nodes: List[int]):
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            for i in range(self.n_estimators):
                futures.append(executor.submit(self.trees[i].update_tree,
                                               arrival_nodes[i], samples,
                                               abs(self.current_profile - 1)))
        wait(futures)
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            for i in range(self.n_estimators):
                futures.append(executor.submit(self.trees[i].reset_profile(self.current_profile)))
        wait(futures)
        self.current_profile = abs(self.current_profile - 1)

    def get_terminal_node(self, samples: np.ndarray) -> List[List[int]]:
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            for i in range(self.n_estimators):
                futures.append(executor.submit(self.trees[i].get_terminal_node,
                                               samples, self.current_profile))
        wait(futures)
        arrival_nodes = []
        for future in futures:
            arrival_nodes.append(future.result())
        return arrival_nodes