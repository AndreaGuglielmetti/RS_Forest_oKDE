import numpy as np
from IForest import IForest
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

X = np.zeros((10000))

for i in range(8):
    X += np.random.normal(
        loc= 3 * ((2 / 3)**i - 1),
        scale= (2 / 3) ** (2 * i),
        size= X.shape
    )
X = X.reshape((-1, 1))

forest = IForest(n_estimators=30,
                 max_samples=1000)
forest.fit(X[:1000])

# kde = KernelDensity().fit(X[:500])
scores, nodes = forest.score(X[1000:1500])


print(scores)
