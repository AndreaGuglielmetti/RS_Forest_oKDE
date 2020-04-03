from Forest.RSForest.RSTree.RSTree_numpy import RSTreeArrayBased
import numpy as np


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

def test_creation(max_depth, node_size_limit):
    tree = RSTreeArrayBased(max_depth=max_depth, node_size_limit=node_size_limit)
    assert tree.node_count == 2 ** (max_depth + 1) - 1
    assert tree.first_leaf_index == 2 ** max_depth - 1
    assert tree.not_eval_nodes_info == {}
    assert np.all(tree.size == 0)
    assert np.all(tree.log_scaled_ratio == -np.inf)
    assert np.all(tree.split_value == -np.inf)
    assert np.all(tree.split_attr == -1)
    return tree

def test_node_building(tree: RSTreeArrayBased):
    x = np.random.standard_normal(size=10000)
    y = x + 3 * np.random.standard_normal(size=10000)
    samples = np.hstack((
        x.reshape((-1, 1)),
        y.reshape((-1, 1))
    ))
    bounds = _compute_bounds(samples, True)
    tree = tree.fit(bounds, samples)
    while tree.not_eval_nodes_info != {}:
        node_id = list(tree.not_eval_nodes_info.keys())[0]
        tree._build_node(node_id)
        if node_id < tree.first_leaf_index:
            assert (2 * node_id + 1) in tree.not_eval_nodes_info.keys()
            assert (2 *  node_id + 2) in tree.not_eval_nodes_info.keys()
    assert tree.not_eval_nodes_info == {}
    assert np.all(tree.log_scaled_ratio != -np.inf), np.where(tree.log_scaled_ratio == -np.inf)
    for i in range(tree.max_depth):
        assert np.isclose(np.sum(np.exp(tree.log_scaled_ratio[2**i - 1: 2**(i+1)-1])), 1), f'Level {i}'


if __name__ == '__main__':
    for i in range(20):
        tree = test_creation(10, 20)
        test_node_building(tree)
        print('Everything passed')