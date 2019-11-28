from Forest.RSForest.RSForest import RSForest
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
import time

x: np.ndarray = np.random.normal(size=2000)
y: np.ndarray = x * 3 + np.random.normal(size=2000)

nbins = 300
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()

samples = np.hstack((
    x.reshape((-1, 1)),
    y.reshape((-1, 1))
))

forest = RSForest(n_estimators=10, max_depth=5, max_samples=256, max_node_size=.1)
start = time.time()
forest.fit(samples, enlarge_bounds=True)
print(time.time() - start)
start = time.time()
scores = forest.score(
    np.hstack((
        xi.flatten().reshape((-1, 1)),
        yi.flatten().reshape((-1, 1))
    ))
)
print(time.time()-start)
plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
plt.show()
#
# x: np.ndarray = np.random.normal(size=256)
# y: np.ndarray = x * 3 + np.random.normal(size=256)
# samples = np.hstack((
#     x.reshape((-1, 1)),
#     y.reshape((-1, 1))
# ))
#
# arr_nodes = forest.get_terminal_node(samples)
# forest.update_forest(samples, arr_nodes)
# scores = forest.score(
#     np.hstack((
#         xi.flatten().reshape((-1, 1)),
#         yi.flatten().reshape((-1, 1))
#     ))
# )
# plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
# plt.show()
