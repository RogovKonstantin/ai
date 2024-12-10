import matplotlib.pyplot as plt
import numpy as np


def plot(X, Y, theta, cars=None, predicted_profit=None):
    # Построение исходных точек данных
    plt.scatter(X, Y, marker='X', color='green', label='Данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.title('График зависимости прибыли от количества автомобилей')

    # Построение линии линейной регрессии
    X_line = np.linspace(min(X), max(X), 100)
    Y_line = theta[0] + theta[1] * X_line
    plt.plot(X_line, Y_line, color='blue', label='Линия регрессии')

    # Нанесение красной точки для предсказания
    if cars is not None and predicted_profit is not None:
        plt.scatter(cars, predicted_profit, color='red', s=100, label='Прогноз')

    plt.legend()
    # Сохранение графика как файла
    plt.savefig("plotResult.png")
    plt.show()  # Отобразить график
