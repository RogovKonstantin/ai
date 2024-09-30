import numpy as np
import matplotlib.pyplot as plt


# Функционал ошибки
def compute_error(theta_values, x, y):
    error_values = np.zeros_like(theta_values)
    for i, theta in enumerate(theta_values):
        h_x = theta * x
        error_values[i] = np.sum((h_x - y) ** 2)
    return error_values


# Построение графика зависимости J от theta1
def graph(theta_values, error_values, title, xlabel, ylabel):
    plt.figure(figsize=(10, 10))
    plt.plot(theta_values, error_values, label='J(theta1)', color='blue')
    plt.yticks(np.arange(min(error_values), max(error_values), 10000))
    plt.xticks(np.arange(-5.5, 6, 0.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


x = np.arange(1, 21)
y = x
y_noisy = y + np.random.uniform(-4, 4, size=x.shape)
theta1_values = np.linspace(-7, 7, 100)

# Без шума
error_values = compute_error(theta1_values, x, y)
theta1_min = theta1_values[np.argmin(error_values)]
print(f'Minimum J(theta1) при theta1 = {theta1_min}')
graph(theta1_values, error_values, 'J(theta1) от theta1', 'theta1', 'J(theta1)')

# С шумом
error_noisy_values = compute_error(theta1_values, x, y_noisy)
theta1_min_noisy = theta1_values[np.argmin(error_noisy_values)]
print(f'Minimum J(theta1) при theta1 (зашумленные данные) = {theta1_min_noisy}')
graph(theta1_values, error_noisy_values,
      'J(theta1) от theta1 (зашумленные данные)', 'theta1', 'J(theta1)')
