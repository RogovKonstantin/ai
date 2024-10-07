import numpy as np
import matplotlib.pyplot as plt


def plot_approximating_lines(x, y_clean, y_noisy, theta1_values, title, filename):
    plt.figure(figsize=(10, 6))

    for theta1 in theta1_values:
        h_x_clean = theta1 * x
        h_x_noisy = theta1 * x

        # Прямые для чистых данных (синяя сплошная линия)
        plt.plot(x, h_x_clean, color='blue', alpha=0.7, linewidth=2,
                 label=f'Чистые данные' if theta1 == theta1_values[0] else "")
        # Прямые для зашумленных данных (оранжевая пунктирная линия)
        plt.plot(x, h_x_noisy, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Зашумленные данные' if theta1 == theta1_values[0] else "")

    plt.scatter(x, y_clean, color='green', label='Чистые данные')
    plt.scatter(x, y_noisy, color='red', label='Зашумленные данные')

    plt.xlim(0, 21)
    plt.xticks(np.arange(0, 22, 1))
    plt.ylim(0, max(np.concatenate((y_clean, y_noisy))) * max(theta1_values) + 1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("h(x) = theta1 * x")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)


def plot_error_function(x, y_clean, y_noisy, theta1_values, title, filename):
    J_theta1_clean = [np.sum((theta1 * x - y_clean) ** 2) for theta1 in theta1_values]
    J_theta1_noisy = [np.sum((theta1 * x - y_noisy) ** 2) for theta1 in theta1_values]

    theta1_min_clean = theta1_values[np.argmin(J_theta1_clean)]
    theta1_min_noisy = theta1_values[np.argmin(J_theta1_noisy)]

    plt.figure()
    plt.plot(theta1_values, J_theta1_clean, label='J(theta1) Чистые данные', color='b')
    plt.plot(theta1_values, J_theta1_noisy, label='J(theta1) Зашумленные данные', color='orange')
    plt.scatter(theta1_min_clean, min(J_theta1_clean), color='blue')
    plt.scatter(theta1_min_noisy, min(J_theta1_noisy), color='red')
    plt.ylim([0, 250])
    plt.xlim([0.5, 1.5])
    plt.title(title)
    plt.xlabel('theta1')
    plt.ylabel('J(theta1)')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)

    return theta1_min_clean, theta1_min_noisy


x = np.arange(1, 21)
y_clean = x
y_noisy = x + np.random.uniform(-2, 2, x.shape)

theta1_values_line = np.linspace(0, 2, 5)
theta1_values_error = np.linspace(0, 2, 120)

# Для прямых
plot_approximating_lines(x, y_clean, y_noisy, theta1_values_line,
                         "Аппроксимирующие прямые для чистых и зашумленных данных", 'lines.png')

# Для ошибки
theta1_min_clean, theta1_min_noisy = plot_error_function(x, y_clean, y_noisy, theta1_values_error,
                                                         "Функционал ошибки для чистых и зашумленных данных",
                                                         'error.png')

print(f'Минимум theta1 (Чистые данные): {theta1_min_clean}')
print(f'Минимум theta1 (Зашумленные данные): {theta1_min_noisy}')
