import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape

        random_ids = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_ids]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)

            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.k)]
            )

            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        return labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    @staticmethod
    def compute_wcss(X, labels, centroids):
        wcss = 0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss

    def elbow_method(self, X, max_k=10):
        wcss_values = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(k=k)
            labels = kmeans.fit(X)
            wcss = KMeans.compute_wcss(X, labels, kmeans.centroids)
            wcss_values.append(wcss)

        print(wcss_values)

        plt.plot(range(1, max_k + 1), wcss_values, marker="o")
        plt.xlabel("K")
        plt.ylabel("WCSS")
        plt.show()

    @staticmethod
    def plot_clusters(X, labels, centroids):
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=200)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Original Data ")
    plt.show()

    kmeans = KMeans(k=3, max_iters=100)
    labels = kmeans.fit(X)

    print("Centroids:")
    print(kmeans.centroids)

    KMeans.plot_clusters(X, labels, kmeans.centroids)
    kmeans.elbow_method(X, max_k=8)
