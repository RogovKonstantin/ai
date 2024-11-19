import matplotlib.pyplot as plt
import numpy as np


def plot_data(x, y, theta):
    plt.scatter(x, y, marker='x', color='green', label='Данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.title('График зависимости прибыли от количества автомобилей')

    x_line = np.linspace(min(x), max(x), 100)
    y_line = theta[0] + theta[1] * x_line

    plt.plot(x_line, y_line, color='red', label='Линия регрессии')
    plt.legend()
    plt.savefig("result.png")

