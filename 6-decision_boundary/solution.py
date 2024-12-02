import numpy as np
import matplotlib.pyplot as plt

# Модель: f(x1, x2) = 1 + 2x2 + x1*x2 + x1^2
def model(x1, x2):
    return 1 + 2 * x2 + x1 * x2 + x1**2

# Заданный критерий принятия решения
alpha = 0.5

# Генерация значений для x1
x1 = np.linspace(-3, 3, 500)  # Диапазон x1
x2 = (-1 - x1**2 - 0.5*x1 + alpha) / 2  # Выражение x2 через x1

# Создание сетки для визуализации областей
x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
f_values = model(x1_grid, x2_grid)

# Построение графика
plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, f_values, levels=[-np.inf, alpha, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.6)
plt.contour(x1_grid, x2_grid, f_values, levels=[alpha], colors='black', linewidths=1.5)
plt.plot(x1, x2, label="Разделяющая кривая", color='blue')

# Подписи и оформление
plt.title("Граница разделения классов", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig('result.png')
