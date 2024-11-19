import numpy as np
import compute_cost
import gradient_descent
import warm_up_exercise
import work
import plot_data

size = int(input("Введите размерность матрицы:"))

warm_up_exercise.warmup_exercise_built_in(size)
warm_up_exercise.warmup_exercise_manual(size)

data = np.loadtxt('train_data.txt', delimiter=',')
X = data[:, 0]
Y = data[:, 1]
m = len(Y)

X = np.column_stack((np.ones(m), X))

theta = np.zeros(2)
iterations = 1500
alpha = 0.01



cost_vector = compute_cost.compute_cost_vector(X, Y, theta)
print(f'Значение функции стоимости векторным способом: {cost_vector}')

cost_elements = compute_cost.compute_cost_elements(X, Y, theta)
print(f'Значение функции стоимости поэлементным способом: {cost_elements}')

theta_vector = gradient_descent.gradient_descent_vector(X, Y, theta, alpha, iterations)
print(f'Вектор параметров модели векторным способом: {theta_vector}')

theta_elements = gradient_descent.gradient_descent_elements(X, Y, theta, alpha, iterations)
print(f'Вектор параметров модели поэлементным способом: {theta_elements}')

plot_data.plot_data(X[:, 1], Y, theta_vector)

cars = int(input("Введите количество автомобилей:"))
profit_vector = work.prediction(cars, theta_vector)
profit_elements = work.prediction(cars, theta_elements)

print(f"Прогнозируемая прибыль (векторный способ): {profit_vector}")
print(f"Прогнозируемая прибыль (поэлементный способ): {profit_elements}")


