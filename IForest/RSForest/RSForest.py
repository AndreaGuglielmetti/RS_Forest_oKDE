from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from .RSTree import RSTree

class RSForest:
    trees: list
    n_estimators: int
    max_samples: int
    max_features: int
    current_profile: int

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, max_samples: int = 256, max_features: float = 1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self.current_profile = 0

    def fit(self, samples: np.ndarray, enlarge_bounds: bool = False):
        self.max_samples = min(self.max_samples, samples.shape[0])
        if samples.shape[0] > self.max_samples:
            indices = np.random.choice(range(samples.shape[0]), size=self.max_samples, replace=False)
        else:
            indices = np.arange(self.max_samples)
        bounds = self._compute_bounds(samples, enlarge_bounds)

        with ThreadPoolExecutor(max_workers=self.n_estimators) as executor:
            futures = []
            for i in range(self.n_estimators):
                futures.append(executor.submit(RSTree().fit,
                                               samples, bounds, self.max_depth, self.current_profile))
            wait(futures)
        self.trees = [future.result() for future in futures]

    def _compute_bounds(self, samples: np.ndarray, enlarge: bool):
        if not enlarge:
            return np.apply_along_axis(lambda x: [x.min(), x.max()], axis=0, arr=samples).T
        else:
            means = np.mean(samples, axis=0)
            std_deviations = np.std(samples, axis=0)
            enlarged_lbound = means - 4.645 * std_deviations
            enlarged_ubound = means+ 4.645 * std_deviations
            return np.hstack((
                enlarged_lbound.reshape((-1, 1)),
                enlarged_ubound.reshape((-1, 1))
            ))
