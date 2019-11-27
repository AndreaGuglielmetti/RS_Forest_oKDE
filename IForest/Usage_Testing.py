import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

from IForest.IForest import IForest

x = np.random.normal(size=500)
y = x * 3 + np.random.normal(size=500)

forest = IForest(n_estimators=30, max_samples=500)
forest.fit(
    np.hstack((
        x.reshape((-1, 1)),
        y.reshape((-1, 1))
    ))
)

nbins = 300
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()

zi = forest.score(
    np.hstack((
        xi.flatten().reshape((-1, 1)),
        yi.flatten().reshape((-1, 1))
    ))
)

plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()