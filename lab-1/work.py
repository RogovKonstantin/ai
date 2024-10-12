import numpy as np


def prediction(cars, theta):
    x = np.array([1, cars])
    return x.dot(theta)
