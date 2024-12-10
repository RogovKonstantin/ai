def gradientDescentByVector(X, Y, theta, alpha, iterations):
    m = len(Y)

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - Y
        theta -= (alpha / m) * (X.T.dot(errors)) #theta=theta-a/m * SUM(h(x^i)-y^i)X^ij

    return theta


def gradientDescentByElements(X, Y, theta, alpha, iterations):
    m = len(Y)  # Число примеров
    n = len(theta)  # Число параметров

    for _ in range(iterations):
        # Инициализируем массив градиентов для обновления theta
        gradients = [0] * n

        # Рассчитаем градиенты для всех параметров
        for i in range(m):  # Проходим по каждому примеру
            # Вычисляем линейную комбинацию вручную (prediction)
            prediction = 0
            for j in range(n):
                prediction += theta[j] * X[i][j]

            # Ошибка предсказания
            error = prediction - Y[i]

            # Обновляем градиенты
            for j in range(n):
                gradients[j] += error * X[i][j]

        # Обновляем параметры theta на основе вычисленных градиентов
        for j in range(n):
            theta[j] -= (alpha / m) * gradients[j]

    return theta
