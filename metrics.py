import numpy as np


class metrics:
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_pred == y_true))
        TN = np.sum((y_pred == 0) & (y_pred == y_true))
        FP = np.sum((y_pred == 1) & (y_pred != y_true))
        FN = np.sum((y_pred == 0) & (y_pred != y_true))
        return TP, TN, FP, FN

    @staticmethod
    def accuracy(y_true, y_pred):
        TP, TN, FP, FN = metrics.confusion_matrix(y_true, y_pred)
        return (TP + TN) / (TP + TN + FP + FN)

    @staticmethod
    def recall(y_true, y_pred):
        TP, TN, FP, FN = metrics.confusion_matrix(y_true, y_pred)
        return (TP) / (TP + FN) if (TP + FN) != 0 else 0

    @staticmethod
    def precision(y_true, y_pred):
        TP, TN, FP, FN = metrics.confusion_matrix(y_true, y_pred)
        return (TP) / (TP + FP) if (TP + FP) != 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        p = metrics.precision(y_true, y_pred)
        r = metrics.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if p + r != 0 else 0

    @staticmethod
    def r_square(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)

        return 1 - ss_res / ss_total

    @staticmethod
    def r_square_adjusted(y_true, y_pred, n_features):
        n = len(y_true)
        r2 = metrics.r_square(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
