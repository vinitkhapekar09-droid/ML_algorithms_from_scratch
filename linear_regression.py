import linear_regression_practice as prev
import numpy as np


class LinearRegresssion:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros((n_features, 1))
        self.b = 0

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (2 / n_samples) * np.sum(X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = np.mean(error**2)

            self.loss_history.append(loss)

    def predict(self, X):
        return X @ self.w + self.b


if __name__ == "__main__":
    X, y = prev.generate_data()
    model = LinearRegresssion(lr=0.01, epochs=500)
    model.fit(X, y)

    print("Learned w: ", model.w)
    print("Learned b: ", model.b)
