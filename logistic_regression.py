import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X_Samples, X_features = X.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros((X_features, 1))
        self.b = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z=z)

            dw = (1 / X_Samples) * X.T @ (y_pred - y)
            db = (1 / X_Samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            eps = 1e-9
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )
            self.loss_history.append(loss)

    @staticmethod
    def generate_classification_data(n=200):
        X = np.random.randn(n, 2)

        true_w = np.array([[2], [-3]])
        true_b = 0.5

        logits = X @ true_w + true_b
        probs = 1 / (1 + np.exp(-logits))

        y = (probs > 0.5).astype(int)

        return X, y

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    X, y = LogisticRegression.generate_classification_data()

    model = LogisticRegression(0.1, 1000)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Accuracy: ", accuracy)

    print("Learned w: ", model.w)
    print("Learned b: ", model.b)
