import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 21)
y = x

theta1_values = np.linspace(-7, 7, 100)

# Добавление шума
y_noise = y + np.random.uniform(-3, 3, size=x.shape)

# Функционал ошибки
j_values = []

for theta in theta1_values:
    hypothesis = theta * x
    error_noise = np.sum((hypothesis - y_noise) ** 2)
    j_values.append(error_noise)

#Минимальное значение J
theta1_min = theta1_values[np.argmin(j_values)]
print(f'J достигает минимума при тета = {theta1_min}')

#Построение графика зависимости J от theta1
plt.figure(figsize=(10, 10))
plt.plot(theta1_values, j_values, color='blue')
plt.yticks(np.arange(min(j_values), max(j_values), 10000))
plt.xticks(np.arange(-5.5, 6, 0.5))
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.grid(True)
plt.show()


