import matplotlib.pyplot as plt
import random

# Координаты точек
naughts = [(-1, 1.1), (-0.7, 0.8), (-0.4, 1.2), (-0.1, 0.1), (0.3, 0.9)]
crosses = [(0.65, -1.45), (0.7, -0.40), (0.95, -0.9), (1.35, -0.2), (1.37, -0.9)]

# Добавляем фиктивный столбец x0 = 1
data = [(1, x1, x2, 1) for x1, x2 in crosses] + [(1, x1, x2, 0) for x1, x2 in naughts]

# Инициализация весов
weights = [random.uniform(-1, 1) for _ in range(3)]


# Функция для обучения персептрона
def train_perceptron(data, weights):
    iteration = 0
    is_converged = False
    while not is_converged:
        iteration += 1
        print(f"\nИтерация {iteration}")
        print("Класс | x0  | x1    | x2    | w0     | w1     | w2     | z      | y  | Верно?")
        print("-" * 80)

        is_converged = True
        for x0, x1, x2, y_true in data:
            z = weights[0] * x0 + weights[1] * x1 + weights[2] * x2
            y_pred = 1 if z >= 0 else 0
            is_correct = y_pred == y_true

            # Форматирование для отображения класса
            class_label = '+' if y_true == 1 else '-'

            # Вывод текущей строки таблицы
            print(
                f"   {class_label}  | {x0:3} | {x1:5.2f} | {x2:5.2f} | {weights[0]:6.2f} | {weights[1]:6.2f} | {weights[2]:6.2f} | {z:6.2f} |  {y_pred} |  {'+' if is_correct else '-'}")

            if not is_correct:
                is_converged = False
                if y_pred == 0:
                    weights[0] += x0
                    weights[1] += x1
                    weights[2] += x2
                else:
                    weights[0] -= x0
                    weights[1] -= x1
                    weights[2] -= x2

    return weights


# Обучение
weights = train_perceptron(data, weights)


# Построение разделяющей прямой
def plot_decision_boundary(weights):
    x = [-1.5, 2]  # Диапазон для x1
    y = [-(weights[0] + weights[1] * x_val) / weights[2] for x_val in x]  # Расчет x2
    plt.plot(x, y, 'r-', label='Decision Boundary')


# График
plt.figure(figsize=(8, 6))
for x1, x2 in crosses:
    plt.scatter(x1, x2, color='blue', label='Crosses' if x1 == crosses[0][0] else "")
for x1, x2 in naughts:
    plt.scatter(x1, x2, color='green', label='Naughts' if x1 == naughts[0][0] else "")
plot_decision_boundary(weights)

# Установка масштаба 1:1
plt.gca().set_aspect('equal', adjustable='box')

# Добавляем оси
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Включаем сетку
plt.grid(True)

# Добавляем подписи и легенду
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Perceptron Decision Boundary')

# Показываем график
plt.show()
