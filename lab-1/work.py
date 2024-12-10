import numpy as np


def predict_profit(cars, theta):
    X = np.array([1, cars])
    return X.dot(theta)
