import numpy as np
from math import log
from typing import Dict, List, Tuple


class RSTreeArrayBased:
    children_left: np.ndarray
    children_right: np.ndarray
    split_attr: np.ndarray
    split_value: np.ndarray
    log_scaled_ratio: np.ndarray
    node_size_limit: int
    max_depth: int
    node_count: int
    first_leaf_index: int
    size: np.ndarray

    TREE_LEAF: int = -1
    TREE_UNDEFINED: int = -2

    def __init__(self, max_depth: int, node_size_limit: int):
        self.max_depth = max_depth
        self.node_size_limit = node_size_limit
        self.node_count = 2 ** (max_depth + 1) - 1
        self.first_leaf_index = 2 ** max_depth - 1
        self.children_left = self.TREE_LEAF * np.ones(self.first_leaf_index, dtype=int)
        self.children_right = self.TREE_LEAF * np.ones(self.first_leaf_index, dtype=int)
        self.split_attr = self.TREE_UNDEFINED * np.ones(self.node_count, dtype=int)
        self.split_value = self.TREE_UNDEFINED * np.ones(self.node_count)
        self.log_scaled_ratio = self.TREE_UNDEFINED * np.ones(self.node_count)
        self.size = np.zeros((2, self.node_count), dtype=int)

    def fit(self, bounds: np.ndarray, samples: np.ndarray) -> 'RSTreeArrayBased':
        boundaries = np.empty((self.node_count, bounds.shape[0], 2))
        boundaries[0] = bounds

        valid_index_per_node = {node_id: [] for node_id in range(self.node_count)}
        valid_index_per_node[0] = list(range(samples.shape[0]))

        for i in range(self.node_count):
            self._build_node(i, boundaries)
            self._populate_node(i, samples, valid_index_per_node)
        return self

    def _build_node(self, node_id: int, boundaries: np.ndarray) -> None:
        node_bounds = boundaries[node_id]
        feature_indices = np.arange(node_bounds.shape[0])[np.apply_along_axis(lambda x: not np.isclose(x, x[0]).all(),
                                                                              axis=0, arr=node_bounds)]
        split_attr = np.random.choice(feature_indices)
        random_value = np.random.uniform(1e-100, 1)
        split_value = node_bounds[split_attr, 0] + random_value * (node_bounds[split_attr, 1] -
                                                                   node_bounds[split_attr, 0])

        # common attributes to each node
        self.split_attr[node_id] = split_attr
        self.split_value[node_id] = split_value
        if node_id != 0:
            parent_idx = self._get_parent(node_id)
            self.log_scaled_ratio[node_id] = self.log_scaled_ratio[parent_idx] + log(random_value)
        else:
            self.log_scaled_ratio[node_id] = 0

        # if the current node is a leaf do not create children
        if node_id < self.first_leaf_index:
            left_children_id = 2 * node_id + 1
            right_children_id = 2 * node_id + 2
            self.children_left[node_id] = left_children_id
            self.children_right[node_id] = right_children_id
            boundaries[left_children_id] = boundaries[node_id]
            boundaries[left_children_id, split_attr, 1] = split_value
            boundaries[right_children_id] = boundaries[node_id]
            boundaries[right_children_id, split_attr, 0] = split_value

    @staticmethod
    def _get_parent(node_id: int) -> int:
        if node_id % 2 == 0:
            parent_idx = (node_id - 2) / 2
        else:
            parent_idx = (node_id - 1) / 2
        return int(parent_idx)

    def _populate_node(self, node_id: int, samples: np.ndarray, valid_index_per_node: Dict[int, List[int]]) -> None:
        valid_indexes = valid_index_per_node[node_id]
        self.size[0, node_id] = len(valid_indexes)

        if node_id < self.first_leaf_index:
            valid_index_per_node[self.children_left[node_id]] = [valid_indexes[i] for i in np.where(
                samples[valid_indexes, self.split_attr[node_id]] <= self.split_value[node_id])[0]]
            valid_index_per_node[self.children_right[node_id]] = [valid_indexes[i] for i in np.where(
                samples[valid_indexes, self.split_attr[node_id]] > self.split_value[node_id])[0]]

    def _navigate_tree_down(self, sample: np.ndarray, current_profile: int, starting_node: int = 0,
                            depth: int = -1) -> int:
        current_node = starting_node
        depth_counter = 0
        while True:
            if depth_counter <= depth:
                return current_node
            if self.size[current_profile, current_node] <= self.node_size_limit or \
                    current_node >= self.first_leaf_index:
                return current_node
            elif sample[self.split_attr[current_node]] <= self.split_value[current_node]:
                current_node = self.children_left[current_node]
            else:
                current_node = self.children_right[current_node]
            depth_counter += 1

    def _navigate_tree_up(self, starting_node: int = 0, depth: int = -1) -> int:
        if depth == -1 or starting_node == 0:
            return 0
        else:
            current_node = starting_node
            for _ in range(depth):
                current_node = self._get_parent(current_node)
            return current_node

    def score(self, samples: np.ndarray, current_profile: int) -> Tuple[np.ndarray, np.ndarray]:
        terminal_size = np.empty(samples.shape[0])
        log_scaled_ratio = np.empty(samples.shape[0])
        for i, sample in enumerate(samples):
            terminal_node = self._navigate_tree_down(sample, current_profile)
            terminal_size[i] = self.size[current_profile, terminal_node]
            log_scaled_ratio[i] = self.log_scaled_ratio[terminal_node]
        return terminal_size, log_scaled_ratio

    def get_terminal_node(self, samples: np.ndarray, current_profile: int) -> List[int]:
        return [self._navigate_tree_down(sample, current_profile) for sample in samples]

    def update_tree(self, terminal_nodes: List[int], samples: np.ndarray, profile_to_update: int,
                    is_anomaly: List[bool] = None) -> None:
        if is_anomaly is None:
            for i, sample in enumerate(samples):
                self._update_tree(terminal_nodes[i], sample, profile_to_update, False)

    def _update_tree(self, terminal_node: int, sample: np.ndarray, profile: int, is_anomaly: bool) -> None:
        if not is_anomaly:
            current_node_size = self.size[profile, terminal_node]
            if current_node_size > self.node_size_limit and terminal_node < self.first_leaf_index:
                child_node = self._navigate_tree_down(sample, profile, starting_node=terminal_node, depth=1)
                while child_node < self.first_leaf_index:
                    self.size[profile, child_node] += 1
                    child_node = self._navigate_tree_down(sample, profile, starting_node=child_node, depth=1)
        else:
            ancestor = terminal_node
            while ancestor != 0:
                self.size[profile, ancestor] -= 1
                ancestor = self._get_parent(ancestor)
