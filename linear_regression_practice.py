import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=100):
    X = np.random.randn(n, 1)
    true_w = 3.0
    true_b = 2.0
    noise = np.random.randn(n, 1) * 0.1
    y = true_w * X + true_b + noise
    return X, y


def predict(X, w, b):
    return X * w + b


def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def gradient_compute(X, y, y_pred):
    n = len(X)

    dw = (2 / n) * np.sum(X * (y_pred - y))
    db = (2 / n) * np.sum(y_pred - y)
    return dw, db


def train(X, y, lr=0.01, epochs=100):
    w = 0.0
    b = 0.0
    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)

        dw, db = gradient_compute(X, y, y_pred)

        w -= lr * dw
        b -= lr * db

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    return w, b


if __name__ == "__main__":
    lr = 0.1
    epochs = 100
    X, y = generate_data()
    w, b = train(X, y, lr, epochs)

    print("Learned w:", w)
    print("Learned b:", b)

    # X, y = generate_data()
    # print(predict(X[:5], 1.2, 0.2))

    # plt.scatter(X, y)

    y_pred = predict(X, w, b)
    print(mse_loss(y, y_pred))

    # plt.scatter(X, y_pred)

    plt.scatter(X, y, label="True")
    plt.scatter(X, y_pred, label="Pred")
    plt.legend()
    plt.show()
