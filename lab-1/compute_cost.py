import numpy as np


def compute_cost_vector(x, y, theta):
    m = len(y)
    predictions = np.dot(x, theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost


def compute_cost_elements(X, Y, theta):
    m = len(Y)
    total_cost = 0

    for i in range(m):
        prediction = 0

        for j in range(len(theta)):
            prediction += theta[j] * X[i][j]
        error = 0
        for j in range(len(theta)):
            error += (theta[j] * X[i][j])

        error -= Y[i]
        total_cost += error ** 2

    cost = (1 / (2 * m)) * total_cost
    return cost


