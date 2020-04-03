
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
import time

from Forest import RSForest

x = np.random.standard_normal(size=10000)
y = np.random.standard_normal(size=10000)

nbins = 100
start = time.time()
k = kde.gaussian_kde([x, y])
print(time.time() - start)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
start = time.time()
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
print(time.time() - start)
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.colorbar()
plt.show()

samples = np.hstack((
    x.reshape((-1, 1)),
    y.reshape((-1, 1))
))

forest = RSForest(n_estimators=100, max_depth=15, max_samples=10000, max_node_size=.01)
start = time.time()
forest.fit(samples, enlarge_bounds=True)
print(f'Fitted {forest.max_samples} samples in {time.time() - start}')
start = time.time()
scores = forest.score_samples(
    np.hstack((
        xi.flatten().reshape((-1, 1)),
        yi.flatten().reshape((-1, 1))
    )),
    normalize=True
)
print(f'Scored {xi.size} samples in {time.time() - start}')
plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
plt.colorbar()
plt.show()
# #
for _ in range(20):
    x: np.ndarray = np.random.standard_normal(size=10000)
    y: np.ndarray = np.random.standard_normal(size=10000)
    samples = np.hstack((
        x.reshape((-1, 1)),
        y.reshape((-1, 1))
    ))

    start = time.time()
    forest.update_forest(samples)
    print(f'Model updated in {time.time() - start}')
    start = time.time()
    scores = forest.score_samples(
        np.hstack((
            xi.flatten().reshape((-1, 1)),
            yi.flatten().reshape((-1, 1))
        )),
        normalize=True
    )
    print(f'Scored {xi.size} samples in {time.time() - start}')
    plt.pcolormesh(xi, yi, scores.reshape(xi.shape))
    plt.colorbar()
    plt.show()
    print()
