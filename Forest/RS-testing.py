from Forest.RSForest.RSForest import RSForest
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
import time

x: np.ndarray = np.random.standard_normal(size=10000)
y: np.ndarray = np.random.standard_normal(size=10000)

nbins = 100
# start = time.time()
# k = kde.gaussian_kde([x, y])
# print(time.time() - start)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
# start = time.time()
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
# print(time.time() - start)
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
# plt.colorbar()
# plt.show()

samples = np.hstack((
    x.reshape((-1, 1)),
    y.reshape((-1, 1))
))

forest = RSForest(n_estimators=1, max_depth=7, max_samples=10000, max_node_size=.01)
start = time.time()
forest.fit(samples, enlarge_bounds=True)
print(time.time() - start)
start = time.time()
scores = forest.score(
    np.hstack((
        xi.flatten().reshape((-1, 1)),
        yi.flatten().reshape((-1, 1))
    )),
    normalize=True
)
print(time.time()-start)
plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
plt.colorbar()
plt.show()
#
for _ in range(5):
    x: np.ndarray = np.random.standard_normal(size=10000)
    y: np.ndarray = np.random.standard_normal(size=10000)
    samples = np.hstack((
        x.reshape((-1, 1)),
        y.reshape((-1, 1))
    ))

    # forest_test = RSForest(n_estimators=1, max_depth=7, max_samples=256, max_node_size=.1)
    # forest_test.fit(samples, enlarge_bounds=True)
    #
    # scores = forest_test.score(
    #     np.hstack((
    #         xi.flatten().reshape((-1, 1)),
    #         yi.flatten().reshape((-1, 1))
    #     )),
    #     normalize=True
    # )
    # plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
    # plt.colorbar()
    # plt.show()

    arr_nodes = forest.get_terminal_node(samples)
    forest.update_forest(samples, arr_nodes)
    scores = forest.score(
        np.hstack((
            xi.flatten().reshape((-1, 1)),
            yi.flatten().reshape((-1, 1))
        )),
        normalize=True
    )
    plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
    plt.colorbar()
    plt.show()
    print()
