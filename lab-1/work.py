import numpy as np


def prediction(cars, theta):
    x = np.array([1, cars])  # Создаем вектор признаков с добавленным свободным членом
    return x.dot(theta)  # Возвращаем предсказание, произведя умножение вектора признаков на параметры модели