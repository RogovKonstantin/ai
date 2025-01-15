import math

# Функция для загрузки набора данных вручную
def load_dataset(file_path):
    """
    Читает CSV-файл построчно и преобразует значения в числа с плавающей точкой.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    headers = lines[0].strip().split(',')
    data = [list(map(float, line.strip().split(','))) for line in lines[1:]]
    return headers, data

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def train_manual(X, y, epochs=2000, learning_rate=0.05):
    """
    Выполняет градиентный спуск для логистической регрессии вручную.
    Возвращает:
        weights: итоговые веса (смещение + коэффициенты)
        errors: список значений MSE на каждую эпоху
        accuracies: список точностей на каждую эпоху
    """
    # Добавляем столбец смещения: форма -> (n_samples, n_features+1)
    X = [[1.0] + x for x in X]

    # Инициализация весов
    weights = [0.0] * len(X[0])

    errors = []
    accuracies = []

    for epoch in range(epochs):
        # Прямой проход
        predictions = []
        for row in X:
            z = sum(w * x for w, x in zip(weights, row))
            predictions.append(sigmoid(z))

        # MSE для отслеживания
        error = sum((p - t) ** 2 for p, t in zip(predictions, y)) / len(y)
        errors.append(error)

        # Точность
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]
        accuracy = sum(1 for pl, t in zip(predicted_labels, y) if pl == t) / len(y)
        accuracies.append(accuracy)

        # Расчет градиента
        for j in range(len(weights)):
            grad = sum((p - t) * row[j] for p, t, row in zip(predictions, y, X)) / len(y)
            weights[j] -= learning_rate * grad

    return weights, errors, accuracies

def evaluate_manual(weights, X, y):
    """
    Оценивает финальные метрики, используя полученные веса.
    Возвращает словарь с метриками: точность, полнота, F1-мера.
    """
    # Добавляем столбец смещения
    X = [[1.0] + x for x in X]

    # Прогнозы
    predictions = []
    for row in X:
        z = sum(w * x for w, x in zip(weights, row))
        predictions.append(sigmoid(z))

    predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

    # Расчет метрик вручную
    tp = sum(1 for pl, t in zip(predicted_labels, y) if pl == 1 and t == 1)
    fp = sum(1 for pl, t in zip(predicted_labels, y) if pl == 1 and t == 0)
    fn = sum(1 for pl, t in zip(predicted_labels, y) if pl == 0 and t == 1)
    tn = sum(1 for pl, t in zip(predicted_labels, y) if pl == 0 and t == 0)

    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Точность": accuracy,
        "Полнота": recall,
        "F1-мера": f1_score,
        "Прецизионность": precision
    }

if __name__ == "__main__":
    dataset_path = '../../datasets/normalized_shuffled_train.csv'
    weights_save_path = '../model_manual_weights.txt'

    EPOCHS = 2000
    LR = 0.05

    # Шаг 1: Загрузка набора данных
    headers, data = load_dataset(dataset_path)

    # Шаг 2: Определение индексов признаков и целевой переменной
    target_index = headers.index('TenYearCHD')
    feature_indices = [i for i in range(len(headers)) if i != target_index]

    # Подготовка X и y
    X_train = [[row[i] for i in feature_indices] for row in data]
    y_train = [row[target_index] for row in data]

    # Шаг 3: Обучение модели (градиентный спуск)
    print("Начало обучения модели с использованием градиентного спуска...")
    weights, errors, accuracies = train_manual(X_train, y_train, epochs=EPOCHS, learning_rate=LR)

    # Шаг 4: Оценка финальной модели
    metrics = evaluate_manual(weights, X_train, y_train)
    print("Итоговые метрики обучения:", metrics)

    # Шаг 5: Сохранение весов модели (смещение + коэффициенты)
    with open(weights_save_path, 'w') as f:
        f.write(','.join(map(str, weights)))
    print(f"Веса модели сохранены в: {weights_save_path}")
    print("Обучение завершено.")
