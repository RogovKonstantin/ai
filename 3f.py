import numpy as np
import matplotlib.pyplot as plt


def plot_approximating_lines(x, y, theta1_values, title, filename):
    plt.figure(figsize=(10, 6))
    for theta1 in theta1_values:
        h_x = theta1 * x  # Аппроксимирующая прямая
        plt.plot(x, h_x, color='blue', alpha=0.2)

    plt.scatter(x, y, color='green')
    plt.xlim(0, 21)
    plt.xticks(np.arange(0, 22, 1))
    plt.ylim(0, max(y) * max(theta1_values) + 1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("h(x) = theta1 * x")
    plt.grid(True)
    plt.savefig(filename)


def plot_error_function(x, y, theta1_values, title, filename):
    J_theta1 = [np.sum((theta1 * x - y) ** 2) for theta1 in theta1_values]
    theta1_min = theta1_values[np.argmin(J_theta1)]

    plt.figure()
    plt.plot(theta1_values, J_theta1, label='J(theta1)', color='b')
    plt.scatter(theta1_min, min(J_theta1), color='red')
    plt.title(title)
    plt.xlabel('theta1')
    plt.ylabel('J(theta1)')
    plt.grid(True)
    plt.savefig(filename)

    return theta1_min


x = np.arange(1, 21)
y_clean = x
y_noisy = x + np.random.uniform(-2, 2, x.shape)

# Для графика
theta1_values_line = np.linspace(0, 2, 5)
# Для функционал ошибки
theta1_values_error = np.linspace(0, 2, 100)

# График для чистых данных
plot_approximating_lines(x, y_clean, theta1_values_line, "Аппроксимирующие прямые(чистые данные)",  '1-1.png')

# Функионал ошибки для чистых данных
theta1_min_clean = plot_error_function(x, y_clean, theta1_values_error, "Функционал ошибки J(theta1) (Чистые данные)",
                                       '1-2.png')
print(f'Минимум theta1 (Чистые данные): {theta1_min_clean}')

# График для зашумленных данных
plot_approximating_lines(x, y_noisy, theta1_values_line,
                         "Аппроксимирующие прямые(зашумленные данные)",
                          '2-1.png')

# Функционал ошибки для зашумленных данных
theta1_min_noisy = plot_error_function(x, y_noisy, theta1_values_error,
                                       "Функционал ошибки J(theta1) (Зашумленные данные)", '2-2.png')
print(f'Минимум theta1 (Зашумленные данные): {theta1_min_noisy}')
