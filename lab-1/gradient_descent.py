import numpy as np


def gradient_descent_vector(x, y, theta, alpha, iterations):
    m = len(y)  # Количество примеров

    # Цикл для выполнения градиентного спуска заданное количество итераций
    for _ in range(iterations):
        predictions = x.dot(theta)  # Вычисляем предсказания модели
        errors = predictions - y  # Вычисляем ошибку
        theta -= (alpha / m) * (x.T.dot(errors))  # Обновляем параметры модели

    return theta  # Возвращаем обновленные параметры модели

# Градиентный спуск поэлементным способом
def gradient_descent_elements(X, y, theta, alpha, iterations):
    m = len(y)  # Количество примеров
    n = len(theta)  # Количество параметров модели

    # Цикл для выполнения градиентного спуска заданное количество итераций
    for _ in range(iterations):
        temp_theta = np.copy(theta)  # Создаем копию параметров модели для обновлений
        # Обновляем каждый параметр модели по очереди
        for j in range(n):
            sum_error = 0
            # Суммируем ошибки для каждого параметра
            for i in range(m):
                prediction = 0
                # Вычисляем предсказание для каждого примера
                for k in range(0, n):
                    prediction += theta[k] * X[i][k]
                error = prediction - y[i]  # Ошибка для данного примера
                sum_error += error * X[i][j - 1] if j > 0 else error  # Обновляем сумму ошибок
            # Обновляем параметр
            temp_theta[j] -= (alpha / m) * sum_error
        theta = temp_theta  # Обновляем параметры модели

    return theta  # Возвращаем обновленные параметры модели