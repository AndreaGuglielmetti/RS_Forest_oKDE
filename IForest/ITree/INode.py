# Import libraries
from concurrent.futures import (ThreadPoolExecutor,
                                wait)
import numpy as np


# INode class
class INode:
    leaf: bool
    size: np.ndarray
    boundaries: np.ndarray
    splitAtt: int
    splitValue: int
    logScaledRatio: float
    parent: 'INode'

    # Fit the node and eventually create its children
    def fit(self, X: np.ndarray, e: int, l: int, fw: np.ndarray, parent: 'INode' = None, prev_random_value: float = 1.):
        self.leaf = True
        self.size = np.ndarray([X.shape[0], 0])
        self.boundaries = np.apply_along_axis(lambda x: (x.min(), x.max()), axis=0, arr=X).T
        self.parent = parent
        if parent is not None:
            self.logScaledRatio = parent.logScaledRatio + np.log(prev_random_value)
        else:
            self.logScaledRatio = 0.
        # Create node children if the conditions are satisfied
        if e < l and X.shape[0] > 1 and not np.isclose(X, X[0]).all():
            self.leaf = False
            # Keep only indices of columns that not have all values identical
            indices = np.asarray(range(X.shape[1]))[np.apply_along_axis(lambda x: not np.isclose(x, x[0]).all(),
                                                                        axis=1, arr=self.boundaries)]
            # Pick up randomly a split attribute and value among the valid ones
            self.splitAtt = np.random.choice(indices, p=fw[indices] / sum(fw[indices]))
            self.splitValue = np.random.uniform(self.boundaries[self.splitAtt][0],
                                                self.boundaries[self.splitAtt][1])
            curr_random_value = self.splitValue - self.boundaries[self.splitAtt][0]
            curr_random_value /=  (self.boundaries[self.splitAtt][1] - self.boundaries[self.splitAtt][0])
            # Build child nodes using multithreading
            futures = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures.append(executor.submit(INode().fit,
                                               X[X[:, self.splitAtt] <= self.splitValue], e + 1, l, fw,
                                               self, curr_random_value))
                futures.append(executor.submit(INode().fit,
                                               X[X[:, self.splitAtt] > self.splitValue], e + 1, l, fw,
                                               self, 1. - curr_random_value))
            wait(futures)
            self.left = futures[0].result()
            self.right = futures[1].result()
        return self

    # Profile the passed sample, returning the depth of the leaf it falls into
    def profile(self, x: np.ndarray, e: int, active_profile: int=0):
        if self.leaf:
            return e + self.c(self.size[active_profile])
        if x[self.splitAtt] <= self.splitValue:
            return self.left.profile(x, e + 1, active_profile)
        else:  # x[self.splitAtt] > self.splitValue
            return self.right.profile(x, e + 1, active_profile)

    def score(self, x: np.ndarray, active_profile: int=0):
        if self.leaf:
            return self.size[active_profile], self.logScaledRatio, self
        elif x[self.splitAtt] <= self.splitValue:
            return self.left.score(x, active_profile)
        else:
            return self.right.score(x, active_profile)
        pass

    def update_path(self, is_anomaly: bool, x: np.ndarray, active_profile: int=0):
        if is_anomaly:
            if self.parent is not None:
                self.parent.size[active_profile] -= 1
                self.parent.update_path(is_anomaly, x, active_profile)
        else:
            if not self.leaf:
                child = self.get_child(x)
                child._update_child(x, active_profile)
            pass
        pass

    def get_child(self, x):
        if self.leaf:
            return None
        elif x[self.splitAtt] <= self.splitValue:
            return self.left
        else:
            return self.right

    def _update_child(self, x, active_profile: int=0):
        self.size[active_profile] += 1
        child = self.get_child(x)
        if child is not None:
            self._update_child(x, active_profile)

    def reset_profile(self, active_profile: int):
        self.size[active_profile] = 0
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures.append(executor.submit(self.left.reset_profile, active_profile))
            futures.append(executor.submit(self.right.reset_profile, active_profile))
        wait(futures)

    @staticmethod
    def c(n: int):
        if n <= 1:
            return 0.
        elif n == 2:
            return 1.
        else:
            return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n
