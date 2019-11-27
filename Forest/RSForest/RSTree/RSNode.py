import numpy as np
from math import log
from concurrent.futures import ThreadPoolExecutor, wait


class RSNode:
    size: np.ndarray
    parent: 'RSNode'
    left: 'RSNode'
    right: 'RSNode'
    log_scaled_ratio: float
    leaf: bool
    split_attr: int
    split_value: float

    def build_structure(self, bounds: np.ndarray, samples: np.ndarray, current_profile: int,
                        curr_depth: int, max_depth: int,
                        parent: 'RSNode' = None, prev_random_value: float = 1.0):
        '''
        Initialize node attributes and recursively generate nodes.
        :param bounds: ndarray, shape(n_features, 2), lower and upper bound for each feature.
        :param curr_depth: int, current depth of the node
        :param max_depth: int, max depth of the tree
        :param parent: RSNode, parent of the node
        :param prev_random_value: float, value in [0, 1] chosen as split value.
        '''
        # creat a leaf
        if curr_depth >= max_depth or samples.shape[0] <= 1:
            self.leaf = True
            self.size = np.zeros(2)
            self.size[current_profile] = samples.shape[0]
            self.parent = parent
            self.log_scaled_ratio = parent.log_scaled_ratio + log(prev_random_value)
            return self
        # creat an internal node and its children
        self.leaf = False
        self.size = np.zeros(2)
        self.size[current_profile] = samples.shape[0]
        indices = np.arange(bounds.shape[0])[np.apply_along_axis(lambda x: not np.isclose(x, x[0]).all(),
                                                                 axis=0, arr=bounds)]
        self.split_attr = np.random.choice(indices)
        random_value = np.random.uniform(1e-100, 1)
        self.split_value = bounds[self.split_attr, 0] + random_value * (bounds[self.split_attr, 1] -
                                                                        bounds[self.split_attr, 0])
        left_bounds = bounds.copy()
        left_bounds[self.split_attr, 1] = self.split_value
        right_bounds = bounds.copy()
        right_bounds[self.split_attr, 0] = self.split_value

        if parent is not None:
            self.log_scaled_ratio = parent.log_scaled_ratio + log(prev_random_value)
        else:
            self.log_scaled_ratio = 0.

        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures.append(executor.submit(RSNode().build_structure,
                                           left_bounds, samples[samples[:, self.split_attr] <= self.split_value],
                                           current_profile,
                                           curr_depth + 1, max_depth,
                                           self, random_value))
            futures.append(executor.submit(RSNode().build_structure,
                                           right_bounds,
                                           samples[samples[:, self.split_attr] > self.split_value],
                                           current_profile,
                                           curr_depth + 1, max_depth,
                                           self, 1 - random_value))
        wait(futures)
        self.left = futures[0].result()
        self.right = futures[1].result()
        return self

    def populate_tree(self, samples: np.ndarray, current_profile: int):
        self.size[current_profile] = samples.shape[0]
        if self.leaf:
            return self
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures.append(executor.submit(self.left.populate_tree,
                                           samples[samples[:, self.split_attr] <= self.split_value],
                                           current_profile))
            futures.append(executor.submit(self.right.populate_tree,
                                           samples[samples[:, self.split_attr] > self.split_value],
                                           current_profile))
        wait(futures)
        return self

    def update(self, sample: np.ndarray, current_profile: int, is_anomaly: bool):
        if not is_anomaly:

            if not self.leaf:
                child = self.get_child(sample)
                child._update_child(sample, current_profile)
        else:
            if self.parent is not None:
                self.parent.size[current_profile] -= 1
                self.parent._update_parent(current_profile)

    def get_child(self, x) -> 'RSNode':
        if self.leaf:
            return None
        elif x[self.split_attr] <= self.split_value:
            return self.left
        else:
            return self.right

    def _update_parent(self, current_profile):
        if self.parent is not None:
            self.parent.size[current_profile] -= 1
            self.parent._update_parent()

    def _update_child(self, sample: np.ndarray, current_profile: int):
        self.size[current_profile] += 1
        child = self.get_child(sample)
        if child is not None:
            self._update_child(sample, current_profile)

    def reset_profile(self, current_profile: int):
        self.size[current_profile] = 0
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures.append(executor.submit(self.left.reset_profile, current_profile))
            futures.append(executor.submit(self.right.reset_profile, current_profile))
        wait(futures)

    def get_terminal_node(self, sample: np.ndarray):
        if self.leaf:
            return self
        elif sample[self.split_attr] <= self.split_value:
            return self.left.get_terminal_node(sample)
        else:
            return self.right.get_terminal_node(sample)

    def score(self, sample: np.ndarray, current_profile: int):
        if self.leaf:
            return self.size[current_profile], self.log_scaled_ratio
        elif sample[self.split_attr] <= self.split_value:
            return self.left.score(sample, current_profile)
        else:
            return self.right.score(sample, current_profile)
