import numpy as np


def compute_cost_vector(x, y, theta):
    m = len(y)
    predictions = x.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost


def compute_cost_elements(x, y, theta):
    m = len(y)
    total_cost = 0

    for i in range(m):
        prediction = x[i].dot(theta)
        error = prediction - y[i]
        total_cost += error ** 2

    cost = (1 / (2 * m)) * total_cost
    return cost
