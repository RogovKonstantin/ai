import numpy as np
import compute_cost
import gradient_descent
import warm_up_exercise
import work
import plot_data

#size = int(input("Введите размерность матрицы:"))
#warm_up_exercise.warmup_exercise_built_in(size)
#warm_up_exercise.warmup_exercise_manual(size)

data = np.loadtxt('train_data.txt', delimiter=',')  # Загружаем данные из текстового файла
X = data[:, 0]  # Признаки (количество автомобилей)
Y = data[:, 1]  # Целевые значения (прибыль)
m = len(Y)  # Количество примеров в обучающей выборке

# Добавляем столбец единиц в матрицу признаков для свободного члена
X = np.column_stack((np.ones(m), X))

# Инициализируем параметры модели
theta = np.zeros(2)  # Начальные параметры модели (ноль)
iterations = 1500  # Количество итераций для градиентного спуска
alpha = 0.01  # Скорость обучения

# Вычисляем стоимость векторным способом
cost_vector = compute_cost.compute_cost_vector(X, Y, theta)
print(f'Значение функции стоимости векторным способом: {cost_vector}')

# Вычисляем стоимость поэлементным способом
cost_elements = compute_cost.compute_cost_elements(X, Y, theta)
print(f'Значение функции стоимости поэлементным способом: {cost_elements}')

# Выполняем градиентный спуск для нахождения параметров модели векторным способом
theta_vector = gradient_descent.gradient_descent_vector(X, Y, theta, alpha, iterations)
print(f'Вектор параметров модели векторным способом: {theta_vector}')

# Выполняем градиентный спуск для нахождения параметров модели поэлементным способом
theta_elements = gradient_descent.gradient_descent_elements(X, Y, theta, alpha, iterations)
print(f'Вектор параметров модели поэлементным способом: {theta_elements}')

# Строим график зависимости прибыли от количества автомобилей с полученными параметрами
plot_data.plot_data(X[:, 1], Y, theta_vector)

# Прогнозируем прибыль для заданного количества автомобилей
cars = int(input("Введите количество автомобилей:"))
profit_vector = work.prediction(cars, theta_vector)  # Прогноз по векторному способу
profit_elements = work.prediction(cars, theta_elements)  # Прогноз по поэлементному способу

print(f"Прогнозируемая прибыль (векторный способ): {profit_vector}")
print(f"Прогнозируемая прибыль (поэлементный способ): {profit_elements}")


