import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n_samples, n_features = X.shape

        # random centroids
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, clusters)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, clusters):
        return np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])
