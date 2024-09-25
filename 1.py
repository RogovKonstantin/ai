import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 21)
y = x

theta1_values = np.linspace(-7, 7, 100)
j_values = []

#Функционал ошибки
for theta in theta1_values:
    hypothesis = theta * x
    error = np.sum((hypothesis - y) ** 2)
    j_values.append(error)

#Минимальное значение J
theta1_min = theta1_values[np.argmin(j_values)]
print(f'J достигает минимума при тета = {theta1_min}')

#Построение графика зависимости J от theta1
plt.figure(figsize=(7, 7))
plt.plot(theta1_values, j_values, color='blue')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.grid(True)
plt.show()


