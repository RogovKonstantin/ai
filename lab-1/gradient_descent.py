import numpy as np


def gradient_descent_vector(x, y, theta, alpha, iterations):
    m = len(y)

    for _ in range(iterations):
        predictions = x.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (x.T.dot(errors))
    return theta


def gradient_descent_elements(X, y, theta, alpha, iterations):
    m = len(y)
    n = len(theta)

    for _ in range(iterations):
        temp_theta = np.copy(theta)
        for j in range(n):
            sum_error = 0
            for i in range(m):
                prediction = 0
                for k in range(0, n):
                    prediction += theta[k] * X[i][k]
                error = prediction - y[i]
                sum_error += error * X[i][j - 1] if j > 0 else error
            temp_theta[j] -= (alpha / m) * sum_error
        theta = temp_theta
    return theta