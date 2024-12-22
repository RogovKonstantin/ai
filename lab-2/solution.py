import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(filepath):
    """Загрузка данных из указанного файла."""
    raw_data = pd.read_csv(filepath, header=None)
    features = raw_data.iloc[:, :-1].values
    target = raw_data.iloc[:, -1].values
    return features, target


def normalize_features(features):
    """Стандартизация признаков путем вычитания среднего и деления на стандартное отклонение."""
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    standardized_features = (features - means) / std_devs
    return standardized_features, means, std_devs


def calculate_cost(features, target, parameters):
    """Расчет стоимости функции для линейной регрессии."""
    num_samples = len(target)
    errors = features @ parameters - target
    cost = (1 / (2 * num_samples)) * np.sum(errors ** 2)
    return cost


def gradient_descent(features, target, parameters, learning_rate, iterations):
    """Градиентный спуск для оптимизации параметров."""
    num_samples = len(target)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        gradient = (features.T @ (features @ parameters - target)) / num_samples
        parameters -= learning_rate * gradient
        cost_history[i] = calculate_cost(features, target, parameters)

    return parameters, cost_history


def solve_normal_equation(features, target):
    """Вычисление параметров линейной регрессии через нормальное уравнение."""
    parameters = np.linalg.pinv(features.T @ features) @ (features.T @ target)
    return parameters


# Загрузка данных
features, target = load_data('ex1data2.txt')
sample_count = len(target)

# Нормализация данных
normalized_features, feature_means, feature_stds = normalize_features(features)

# Добавление единичного столбца для bias
normalized_features = np.c_[np.ones((sample_count, 1)), normalized_features]

# Настройки градиентного спуска
learning_rate = 0.05
max_iterations = 100
initial_parameters = np.zeros(normalized_features.shape[1])

# Запуск градиентного спуска
optimized_parameters, cost_history = gradient_descent(
    normalized_features, target, initial_parameters, learning_rate, max_iterations
)

# Сохранение графика изменения функции стоимости
plt.figure()
plt.plot(range(1, max_iterations + 1), cost_history, '-r', linewidth=2)
plt.xlabel('Итерации')
plt.ylabel('Значение стоимости')
plt.title('Процесс сходимости')
plt.savefig('convergence_plot.png')
plt.close()

# Решение с использованием нормального уравнения
features_with_bias = np.c_[np.ones((sample_count, 1)), features]
normal_parameters = solve_normal_equation(features_with_bias, target)

# Ввод данных пользователем для предсказания
print("\n--- Предсказание стоимости ---")
engine_speed = float(input("Введите скорость двигателя: "))
gear_count = float(input("Введите количество передач: "))

# Нормализация данных пользователя для градиентного спуска
engine_speed_norm = (engine_speed - feature_means[0]) / feature_stds[0]
gear_count_norm = (gear_count - feature_means[1]) / feature_stds[1]
user_input_gd = np.array([[1, engine_speed_norm, gear_count_norm]])

# Предсказания
predicted_gd = user_input_gd @ optimized_parameters
predicted_ne = np.array([[1, engine_speed, gear_count]]) @ normal_parameters

print(f"Предсказанная стоимость (Градиентный спуск): {predicted_gd[0]:.2f}")
print(f"Предсказанная стоимость (Нормальное уравнение): {predicted_ne[0]:.2f}")

# 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(features[:, 0], features[:, 1], target, c='b', label='Фактические значения')
ax.scatter(engine_speed, gear_count, predicted_gd[0], c='r', s=100, label='Предсказание (ГС)')
ax.scatter(engine_speed, gear_count, predicted_ne[0], c='g', s=100, label='Предсказание (НЕ)')

ax.set_xlabel('Скорость двигателя')
ax.set_ylabel('Количество передач')
ax.set_zlabel('Стоимость')
ax.set_title('Сравнение значений')
ax.legend()

plt.savefig('predictions_comparison.png')
plt.close()
