import numpy as np


def gradient_descent_vector(x, y, theta, alpha, iterations):
    m = len(y)

    for _ in range(iterations):
        predictions = x.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (x.T.dot(errors))
    return theta


def gradient_descent_elements(x, y, theta, alpha, iterations):
    m = len(y)

    for _ in range(iterations):
        predictions = np.zeros(m)
        errors = np.zeros(m)

        for i in range(m):
            predictions[i] = x[i].dot(theta)
            errors[i] = predictions[i] - y[i]

        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                gradient += errors[i] * x[i][j]
            theta[j] -= (alpha / m) * gradient

    return theta
