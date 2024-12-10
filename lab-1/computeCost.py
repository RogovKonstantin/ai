import numpy as np


def computeCostByVector(X, Y, theta):
    m = len(Y)
    predictions = np.dot(X, theta) #htheta(x)=theta0+theta1*x1+theta2*x2+...
    errors = predictions - Y #htheta(x)-y
    cost = (1 / (2 * m)) * np.dot(errors, errors) #SUMM(errors^2)
    return cost


def computeCostByElements(X, Y, theta):
    m = len(Y)
    total_cost = 0

    for i in range(m):
        prediction = 0

        for j in range(len(theta)):
            prediction += theta[j] * X[i][j]

        error = prediction - Y[i]
        total_cost += error ** 2

    cost = (1 / (2 * m)) * total_cost
    return cost