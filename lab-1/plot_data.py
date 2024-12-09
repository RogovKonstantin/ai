import matplotlib.pyplot as plt
import numpy as np

def plot_data(x, y, theta):
    plt.scatter(x, y, marker='x', color='green', label='Данные')  # Строим точки данных
    plt.xlabel('Количество автомобилей')  # Подпись по оси X
    plt.ylabel('Прибыль')  # Подпись по оси Y
    plt.title('График зависимости прибыли от количества автомобилей')  # Заголовок графика

    x_line = np.linspace(min(x), max(x), 100)  # Генерируем 100 точек для линии
    y_line = theta[0] + theta[1] * x_line  # Линия регрессии

    plt.plot(x_line, y_line, color='red', label='Линия регрессии')  # Строим линию регрессии
    plt.legend()  # Показываем легенду
    plt.savefig("result.png")  # Сохраняем график в файл