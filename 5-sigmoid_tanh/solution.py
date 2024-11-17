import numpy as np
import matplotlib.pyplot as plt


# Функция для вычисления сигмоиды
def sigmoid_func(x):
    return np.reciprocal(1 + np.exp(-x))


# Производная сигмоиды
def sigmoid_derivative(x):
    s = sigmoid_func(x)
    return s * (1 - s)


# Производная гиперболического тангенса
def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t ** 2


# Вводные данные для расчетов
test_values = [0, 3, -3, 8, -8, 15, -15]

# Подсчет значений для сигмоиды
sigmoid_results = list(map(sigmoid_func, test_values))

# Вывод результатов для сигмоиды
print("Значения функции сигмоида:")
for idx, result in enumerate(sigmoid_results):
    print(f"σ({test_values[idx]}) = {result:.15f}")

# График функции сигмоиды
x_vals = np.linspace(-20, 20, 1000)
y_vals = sigmoid_func(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, color='blue', label="Sigmoid")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.title("График функции сигмоиды")
plt.legend()
plt.grid(True)
plt.savefig('sigmoid_plot.png')

# Графики гиперболических функций
sinh_vals = np.sinh(x_vals)
cosh_vals = np.cosh(x_vals)
tanh_vals = np.tanh(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, sinh_vals, label="sinh(x)", color="red")
plt.plot(x_vals, cosh_vals, label="cosh(x)", color="green")
plt.plot(x_vals, tanh_vals, label="tanh(x)", color="purple")
plt.xlabel("x")
plt.ylabel("Гиперболические функции")
plt.title("Графики гиперболических функций")
plt.legend()
plt.grid(True)
plt.savefig('hyperbolic_functions.png')

# Вычисление производных
sigmoid_derivatives = list(map(sigmoid_derivative, test_values))
tanh_derivatives = list(map(tanh_derivative, test_values))

# Вывод производных
print("\nПроизводные сигмоиды:")
for idx, result in enumerate(sigmoid_derivatives):
    print(f"σ'({test_values[idx]}) = {result:.15f}")

print("\nПроизводные гиперболического тангенса:")
for idx, result in enumerate(tanh_derivatives):
    print(f"tanh'({test_values[idx]}) = {result:.15f}")
