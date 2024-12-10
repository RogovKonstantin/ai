import numpy as np
import computeCost
import gradientDescent
import plotData
import warmUpExercise
import work

# Разминка
# matrixSize = int(input("Введите размерность: "))
# warmUpExercise.warm_up_exercise(matrixSize)

# Загрузка данных
data = np.loadtxt('train_data.txt', delimiter=',')
X = data[:, 0]
Y = data[:, 1]
# Переменная m используется для нормализации функции стоимости (чтобы учитывать количество данных)
m = len(Y)

# Добавление столбца единиц к массиву признаков. Для учета свободного члена в модели линейной регрессии
X = np.column_stack((np.ones(m), X))

theta = np.zeros(2)
iterations = 1500
alpha = 0.01

# Вычисление стоимости (векторное)
costByVector = computeCost.computeCostByVector(X, Y, theta)
print(f'Значение функции стоимости(векторный способ): {costByVector}')

# Вычисление стоимости (поэлементное)
costByElements = computeCost.computeCostByElements(X, Y, theta)
print(f'Значение функции стоимости(поэлементный способ): {costByElements}')

# Градиентный спуск (векторный)
thetaByVector = gradientDescent.gradientDescentByVector(X, Y, theta, alpha, iterations)
print(f'Градиентный спуск(векторный способ): {thetaByVector}')

# Градиентный спуск (поэлементный)
thetaByElements = gradientDescent.gradientDescentByElements(X, Y, theta, alpha, iterations)
print(f'Градиентный спуск(поэлементный способ): {thetaByElements}')

# Количество автомобилей, введенное пользователем
cars = int(input("Введите количество автомобилей: "))

# Вычисление прибыли на основе моделей
profitByVector = work.predict_profit(cars, thetaByVector)
profitByElements = work.predict_profit(cars, thetaByElements)

print(f"Прогнозируемая прибыль СТО (векторный способ): {profitByVector}")
print(f"Прогнозируемая прибыль СТО (поэлементный способ): {profitByElements}")

# Визуализация данных и добавление точки с предсказанным результатом
plotData.plot(X[:, 1], Y, thetaByVector, cars, profitByVector)
