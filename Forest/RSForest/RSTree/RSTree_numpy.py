import numpy as np
from math import log
from typing import Dict, List, Tuple
from queue import SimpleQueue

Q: 'SimpleQueue[int]' = SimpleQueue


class RSTreeArrayBased:
    """
    RS-Tree represented with ndarrays.
    The index of each array is the node-id of each node in the tree.

    Attributes:
        children_left : np.ndarray
            an array containing the index of the left children
        children_right : np.ndarray
            an array containing the index of the right children
        split_attr : np.ndarray
            an array containing the attribute index on which the node splits itself
        split_value : np.ndarray
            an array containing the cutting point value
        log_scaled_ratio : np.ndarray
            an array where each cell contains log(volume of the node / volume the feature space)
        node_size_limit : int
            maximum number of samples for being considered a termination node
        first_leaf_index : int
            value that represents the first index of the leaves of the tree
        size : np.ndarray
            array containing the number of samples contained in that node
        not_eval_nodes_info : Dict[int, Tuple[np.ndarray, float]]
            dictionary containing information of the nodes not yet expanded
            key: node-id of the node to expand
            value: Tuple containing the boundaries of the region represented by the node 
                and the random value chosen for the cutting point    
    """

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
    not_eval_nodes_info: Dict[int, Tuple[np.ndarray, float]]

    TREE_LEAF = -1
    TREE_UNDEFINED = -np.inf

    def __init__(self, max_depth: int, node_size_limit: int):
        self.max_depth = max_depth
        self.node_size_limit = node_size_limit
        self.node_count = 2 ** (max_depth + 1) - 1
        self.first_leaf_index = 2 ** max_depth - 1
        # initialize each array with a reserved value
        self.children_left = np.full(self.first_leaf_index, self.TREE_LEAF, dtype=int)
        self.children_right = np.full(self.first_leaf_index, self.TREE_LEAF, dtype=int)
        self.split_attr = np.full(self.node_count, self.TREE_LEAF, dtype=int)
        self.split_value = np.full(self.node_count, self.TREE_UNDEFINED)
        self.log_scaled_ratio = np.full(self.node_count, self.TREE_UNDEFINED)
        self.size = np.zeros((2, self.node_count), dtype=int)
        self.not_eval_nodes_info = {}

    def fit(self, bounds: np.ndarray, samples: np.ndarray) -> 'RSTreeArrayBased':
        """
        Builds the structure of the tree and populates it in a lazy way.
        Starting from the root, each node to expand is inserted into a queue.
        A node is inserted into the queue only if it contains at least one samples,
        empty nodes are stored into not_eval_nodes_info for a future expansion.
        valid_index_per_node is a dictionary in which
            key: node-id
            value: list containing the indexes of the samples that "fall" in the region of the 
                feature space represented by the node
        
        Arguments:
            bounds : np.ndarray, shape(n_dimensions, 2)
                boundaries of the feature space, bounds[:, 0] lower, bounds[:, 1] upper
            samples : nd.ndarray, shape(n_samples, n_dimensions)
                samples used to populate the tree
        
        Returns: self
        """
        self.not_eval_nodes_info[0] = (bounds, 0)
        valid_index_per_node = {node_id: [] for node_id in range(self.node_count)}
        valid_index_per_node[0] = list(range(samples.shape[0]))
        node_queue = SimpleQueue()
        node_queue.put(0)

        while not node_queue.empty():
            node_id = node_queue.get()
            self._build_node(node_id)
            self._populate_node(node_id, samples, valid_index_per_node, node_queue)
        return self

    def _build_node(self, node_id: int) -> None:
        """
        Builds the node inserting into the arrays the id of the left and right child
        (if not a leaf), the split attribute chosen, the cutting point and the log-scaled ratio.        
        """
        node_bounds, prev_random_value = self.not_eval_nodes_info.pop(node_id)
        feature_indices = np.arange(node_bounds.shape[0])[np.apply_along_axis(lambda x: not np.isclose(x, x[0]).all(),
                                                                              axis=0, arr=node_bounds)]
        split_attr = np.random.choice(feature_indices)
        random_value = np.random.uniform(1e-10, 1)
        split_value = node_bounds[split_attr, 0] + random_value * (node_bounds[split_attr, 1] -
                                                                   node_bounds[split_attr, 0])

        # common attributes to each node
        self.split_attr[node_id] = split_attr
        self.split_value[node_id] = split_value
        if node_id != 0:
            parent_idx = self._get_parent(node_id)
            self.log_scaled_ratio[node_id] = self.log_scaled_ratio[parent_idx] + log(prev_random_value)
        else:
            self.log_scaled_ratio[node_id] = 0

        # if the current node is a leaf do not create children
        if node_id < self.first_leaf_index:
            self._prepare_child_info(node_id, node_bounds, split_attr, split_value, random_value)

    def _prepare_child_info(self, node_id: int, bounds: np.ndarray, split_attr: int, split_value: float,
                            random_value: float) -> None:
        """
        Insert into not_eval_nodes_info the values needed for a possible future expansion of the node.
        """
        left_children_id = 2 * node_id + 1
        right_children_id = 2 * node_id + 2

        self.children_left[node_id] = left_children_id
        self.children_right[node_id] = right_children_id
        left_child_bound = bounds.copy()
        left_child_bound[split_attr, 1] = split_value
        self.not_eval_nodes_info[left_children_id] = (left_child_bound, random_value)

        right_child_bound = bounds.copy()
        right_child_bound[split_attr, 0] = split_value
        self.not_eval_nodes_info[right_children_id] = (right_child_bound, 1 - random_value)

    @staticmethod
    def _get_parent(node_id: int) -> int:
        if node_id <= 2:
            return 0
        if node_id % 2 == 0:
            parent_idx = (node_id - 2) // 2
        else:
            parent_idx = (node_id - 1) // 2
        return parent_idx

    def _populate_node(self, node_id: int, samples: np.ndarray, valid_index_per_node: Dict[int, List[int]],
                       node_queue: Q) -> None:
        valid_indexes = valid_index_per_node[node_id]
        self.size[0, node_id] = len(valid_indexes)
        # evaluate if the node needs children
        if self.size[0, node_id] > self.node_size_limit and node_id < self.first_leaf_index:
            left_valid_indexes = [valid_indexes[i] for i in np.where(
                samples[valid_indexes, self.split_attr[node_id]] <= self.split_value[node_id])[0]]
            valid_index_per_node[self.children_left[node_id]] = left_valid_indexes
            right_valid_indexes = [valid_indexes[i] for i in np.where(
                samples[valid_indexes, self.split_attr[node_id]] > self.split_value[node_id])[0]]
            valid_index_per_node[self.children_right[node_id]] = right_valid_indexes
            node_queue.put(self.children_left[node_id])
            node_queue.put(self.children_right[node_id])

    def _navigate_tree_down(self, sample: np.ndarray, current_profile: int, update_profile: bool) -> int:
        """
        Navigates the tree with a single sample until a leaf node or a termination node is reached.
        If the node is not expanded yet, hence it is contained into not_eval_nodes_info, it removes
        the relative entry from not_eval_nodes_info and calls _build_node.
        
        Arguments:
            sample: np.ndarray, shape(n_dimensions)
            current_profile: int
                profile used for determine when a node can be considered a termination node              
            update_profile: bool
                when True, updates the other profile for avoiding another tree navigation
                when update the RS-Tree.
        
        Returns:
            int -- node-id of the leaf or the termination node
        """        
        current_node = 0
        profile_to_update = abs(current_profile - 1)
        while True:
            if self.log_scaled_ratio[current_node] == self.TREE_UNDEFINED:
                self._build_node(current_node)
            assert self.split_attr[current_node] != self.TREE_LEAF

            if update_profile:
                self.size[profile_to_update, current_node] += 1

            if self.size[current_profile, current_node] <= self.node_size_limit or \
                    current_node >= self.first_leaf_index:
                return current_node
            elif sample[self.split_attr[current_node]] <= self.split_value[current_node]:
                current_node = self.children_left[current_node]
            else:
                current_node = self.children_right[current_node]

    def _navigate_tree_down_multi(self, samples: np.ndarray, current_profile: int,
                                  update_profile: bool) -> List[Tuple[int, np.ndarray]]:
        """
        Navigates the tree considering more samples simultaneously.
        If the node is not expanded yet, hence it is contained into not_eval_nodes_info, it removes
        the relative entry from not_eval_nodes_info and calls _build_node.
        
        Arguments:
            samples: np.ndarray, shape(n_samples, n_dimensions)
            current_profile: int
                profile used for determine when a node can be considered a termination node
            update_profile: bool
                when True, updates the other profile for avoiding another tree navigation
                when update the RS-Tree.
        
        Returns:
            List[Tuple[int, np.ndarray]] --
            each tuple is structured as
                0: node-id of the leaf or termination node
                1: index of samples contained into node-id            
        """                         
        node_queue = SimpleQueue()
        node_queue.put((0, np.arange(samples.shape[0])))
        profile_to_update = abs(current_profile - 1)
        terminal_nodes = []
        while not node_queue.empty():
            current_node, current_samples = node_queue.get()
            if self.log_scaled_ratio[current_node] == self.TREE_UNDEFINED:
                self._build_node(current_node)
            if update_profile:
                self.size[profile_to_update, current_node] += current_samples.shape
            if self.size[
                current_profile, current_node] <= self.node_size_limit or current_node >= self.first_leaf_index:
                terminal_nodes.append((current_node, current_samples))
            else:
                self._enqueue_children(current_node, current_samples, node_queue, samples)
        return terminal_nodes

    def _enqueue_children(self, current_node: int, current_samples: np.ndarray,
                          node_queue: Q, samples: np.ndarray) -> None:
        left_child = self.children_left[current_node]
        left_samples = current_samples[
            samples[current_samples, self.split_attr[current_node]] <= self.split_value[current_node]]
        right_child = self.children_right[current_node]
        right_samples = current_samples[
            samples[current_samples, self.split_attr[current_node]] > self.split_value[current_node]]
        for child, child_samples in zip([left_child, right_child], [left_samples, right_samples]):
            if child in self.not_eval_nodes_info.keys():
                self._build_node(child)
            if child_samples.size > 0:
                node_queue.put((child, child_samples))

    def _navigate_tree_up(self, starting_node: int = 0, depth: int = -1) -> int:
        if depth == -1 or starting_node == 0:
            return 0
        else:
            current_node = starting_node
            for _ in range(depth):
                current_node = self._get_parent(current_node)
            return current_node

    def score(self, samples: np.ndarray, current_profile: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each sample, navigates the tree until a termination node or leaf node is reached.
        Then, it stores the number of samples contained in that node and the log-scaled ratio of that node.
        N.B. since the RS-Forest algorithm operates with the latest window, the number of samples contained
        in each node are referred to the last update.
        
        Returns:
            Tuple[np.ndarray, np.ndarray] --
            tuple[0]: number of elements (referred to the last update) contained in the region 
                represented by the leaf of the termination node.
            tuple[1]: log-scaled ratio of the region represented by the node
        """
        terminal_size = np.empty(samples.shape[0])
        log_scaled_ratio = np.empty(samples.shape[0])
        result = self._navigate_tree_down_multi(samples, current_profile, update_profile=False)
        for node, sample_idxs in result:
            terminal_size[sample_idxs] = self.size[current_profile, node]
            log_scaled_ratio[sample_idxs] = self.log_scaled_ratio[node]
        return terminal_size, log_scaled_ratio

    def get_terminal_node(self, samples: np.ndarray, current_profile: int) -> List[Tuple[int, np.ndarray]]:
        return self._navigate_tree_down_multi(samples, current_profile, update_profile=True)

    def update_tree(self, terminal_nodes: List[Tuple[int, np.ndarray]], samples: np.ndarray,
                    profile_to_update: int, is_anomaly: List[bool] = None) -> None:
        if is_anomaly is None:
            self._update_tree(terminal_nodes, samples, profile_to_update)

    def _update_tree(self, terminal_nodes: List[Tuple[int, np.ndarray]], samples: np.ndarray,
                     profile_to_update: int) -> None:
        """
        Implementation of UpdateModel algorithm described in the paper.
        The update for anomalous samples is not implemented
        
        Arguments:
            terminal_nodes List[Tuple[int, np.ndarray]]
                list returnd from _navigate_tree_down_multi
                tuple[0]: node-id
                tuple[1]: samples "fallen" in that node
            samples np.ndarray, shape(n_samples, n_description)
            profile_to_update: int
                the profile to update
        """             
        for node, sample_idxs in terminal_nodes:
            current_node_size = self.size[profile_to_update, node]
            if current_node_size > self.node_size_limit and node < self.first_leaf_index:
                self.size[profile_to_update, node] -= current_node_size
                self._navigate_tree_down_multi(samples[sample_idxs], profile_to_update, update_profile=True)

    def _update_children(self, starting_node: int, samples: np.ndarray, profile_to_update: int) -> None:
        node_queue = SimpleQueue()
        node_queue.put((starting_node, np.arange(samples.shape[0])))
        while not node_queue.empty():
            current_node, current_samples = node_queue.get()
            self.size[profile_to_update, current_node] += samples
            if current_node < self.first_leaf_index:
                self._enqueue_children(current_node, current_samples, node_queue, samples)

    def reset_profile(self, profile_to_erase: int) -> None:
        self.size[profile_to_erase] = 0
