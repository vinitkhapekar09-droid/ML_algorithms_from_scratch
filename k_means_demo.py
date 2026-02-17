from k_means import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Running demo...")

    X, _ = make_blobs(n_samples=300, centers=3, random_state=0)

    kmeans = KMeans(k=3)
    labels = kmeans.fit(X)

    print("Centroids:", kmeans.centroids)

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color="red")
    plt.title("KMeans Clusters")
    plt.show()
