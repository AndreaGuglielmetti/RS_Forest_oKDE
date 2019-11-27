from Forest.RSForest.RSForest import RSForest
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

x: np.ndarray = np.random.normal(size=2000)
y: np.ndarray = x * 3 + np.random.normal(size=2000)

nbins = 300
# k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
# plt.show()

samples = np.hstack((
    x.reshape((-1, 1)),
    y.reshape((-1, 1))
))

forest = RSForest(n_estimators=20, max_depth=4, max_samples=256)
forest.fit(samples, enlarge_bounds=True)
scores = forest.score(
    np.hstack((
        xi.flatten().reshape((-1, 1)),
        yi.flatten().reshape((-1, 1))
    ))
)
plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
plt.show()

x: np.ndarray = np.random.normal(size=256)
y: np.ndarray = x * 3 + np.random.normal(size=256)
samples = np.hstack((
    x.reshape((-1, 1)),
    y.reshape((-1, 1))
))

arr_nodes = forest.get_terminal_node(samples)
forest.update_forest(samples, arr_nodes)
scores = forest.score(
    np.hstack((
        xi.flatten().reshape((-1, 1)),
        yi.flatten().reshape((-1, 1))
    ))
)
plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
plt.show()
