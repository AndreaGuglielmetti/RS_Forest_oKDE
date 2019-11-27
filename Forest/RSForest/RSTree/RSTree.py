import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from .RSNode import RSNode
from typing import List

class RSTree:
    root: RSNode

    def fit(self, samples: np.ndarray, bounds: np.ndarray, max_depth, current_profile):
        self.root = RSNode()
        self.root = self.root.build_structure(bounds, samples, current_profile, curr_depth=0, max_depth=max_depth, parent=None, prev_random_value=1.)
        # self.root = self.root.populate_tree(samples, current_profile)
        return self

    def score(self, samples: np.ndarray, current_profile: int):
        leaf_size = np.empty(samples.shape[0])
        log_scaled_ratio = np.empty(samples.shape[0])
        for i, sample in enumerate(samples):
            leaf_size[i], log_scaled_ratio[i] = self.root.score(sample, current_profile)
        return leaf_size, log_scaled_ratio

    def get_terminal_node(self, samples: np.ndarray):
        nodes = []
        for sample in samples:
            nodes.append(self.root.get_terminal_node(sample))
        return nodes
        # return [self.root.get_terminal_node(sample) for sample in samples]

    def update_tree(self, arrival_nodes: List[RSNode], samples: np.ndarray,
                    current_profile: int, is_anomaly: List[bool] = None):
        if is_anomaly is None:
            for i, sample in enumerate(samples):
                arrival_nodes[i].update(sample, current_profile, False)
        else:
            for i, sample in enumerate(samples):
                arrival_nodes[i].update(sample, current_profile, is_anomaly[i])

    def reset_profile(self, active_profile):
        self.root.reset_profile(active_profile)

