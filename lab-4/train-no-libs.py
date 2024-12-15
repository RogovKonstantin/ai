import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка набора данных из файла
file_path = 'ex2data1.txt'  # Укажите путь к вашему файлу
data = pd.read_csv(file_path, header=None, names=["Vibration", "Rotation", "Label"])

# Разделение данных на признаки (X) и целевую переменную (y)
X_original = data[["Vibration", "Rotation"]].values
y = data["Label"].values

# Масштабирование признаков для улучшения сходимости градиентного спуска
X_mean = np.mean(X_original, axis=0)
X_std = np.std(X_original, axis=0)
X = (X_original - X_mean) / X_std

# Добавление столбца единиц для учета свободного члена (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Добавляем столбец единиц

# Определение сигмоидной функции для преобразования линейной комбинации в вероятность
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция стоимости (логистическая регрессия)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    # Вычисление средней стоимости по всем примерам
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Реализация градиентного спуска для оптимизации параметров модели
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        # Вычисление градиента
        gradient = (1/m) * (X.T @ (sigmoid(X @ theta) - y))
        # Обновление параметров
        theta -= alpha * gradient
        # Сохранение стоимости для анализа сходимости
        cost_history.append(compute_cost(X, y, theta))
        # Опционально: вывод прогресса
        if (i+1) % 1000 == 0:
            print(f"Итерация {i+1}/{num_iters}, Стоимость: {cost_history[-1]:.4f}")

    return theta, cost_history

# Инициализация параметров модели
m, n = X.shape
initial_theta = np.zeros(n)
alpha = 0.1  # Увеличенная скорость обучения для более быстрого сходжения
num_iters = 5000  # Увеличенное количество итераций для достижения сходимости

# Обучение модели с использованием градиентного спуска
final_theta, cost_history = gradient_descent(X, y, initial_theta, alpha, num_iters)

# Функция для предсказания меток на основе обученных параметров
def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= 0.5).astype(int)

# Предсказание на обучающих данных
y_pred = predict(X, final_theta)
# Оценка точности модели
accuracy = np.mean(y_pred == y)

print(f"Точность: {accuracy:.2f}")

# Визуализация границы принятия решений
plt.figure(figsize=(8, 6))

# Определение диапазона значений для построения сетки в исходных масштабах данных
x_min, x_max = data["Vibration"].min() - 1, data["Vibration"].max() + 1
y_min, y_max = data["Rotation"].min() - 1, data["Rotation"].max() + 1
# Создание сетки точек
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Масштабирование сетки с использованием тех же параметров, что и тренировочные данные
X_grid = np.c_[np.ones((xx.size, 1)),
(xx.ravel() - X_mean[0]) / X_std[0],
(yy.ravel() - X_mean[1]) / X_std[1]]

# Предсказание классов для каждой точки сетки
Z = predict(X_grid, final_theta)
Z = Z.reshape(xx.shape)

# Отображение области, соответствующей каждому классу
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# Отображение исходных данных на графике
for label, color, marker in zip([0, 1], ['red', 'blue'], ['o', 'x']):
    subset = data[data["Label"] == label]
    plt.scatter(subset["Vibration"], subset["Rotation"],
                label=f"Класс {label}", c=color, marker=marker, edgecolors='k')

plt.xlabel("Вибрация")
plt.ylabel("Ротация")
plt.legend()
plt.title("Граница принятия решений")
plt.savefig("manual_decision_boundary.png")  # Сохранение графика границы решений

# Визуализация изменения функции стоимости во время обучения
plt.figure(figsize=(8, 6))
plt.plot(range(len(cost_history)), cost_history, label="Стоимость")
plt.xlabel("Итерации")
plt.ylabel("Стоимость")
plt.title("Сходимость функции стоимости")
plt.legend()
plt.savefig("cost_convergence.png")  # Сохранение графика сходимости стоимости
