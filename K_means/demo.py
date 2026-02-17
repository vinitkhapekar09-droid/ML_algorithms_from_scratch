import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans

# fake data
np.random.seed(0)
X = np.vstack(
    [
        np.random.randn(100, 2) + [5, 5],
        np.random.randn(100, 2) + [-5, -5],
        np.random.randn(100, 2) + [5, -5],
    ]
)

# try different K
wcss = []
for k in range(1, 8):
    model = KMeans(k=k)
    model.fit(X)
    clusters = model.assign_clusters(X)
    loss = sum(
        np.linalg.norm(X[clusters == i] - model.centroids[i]) ** 2 for i in range(k)
    )
    wcss.append(loss)

plt.plot(range(1, 8), wcss)
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
