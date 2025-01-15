import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Функция загрузки набора данных
def load_dataset(file_path):
    """
    Загружает CSV файл с набором данных по указанному пути.
    """
    try:
        dataset = pd.read_csv(file_path)
        print(f"Набор данных успешно загружен: {dataset.shape[0]} строк и {dataset.shape[1]} столбцов.")
        return dataset
    except Exception as e:
        print(f"Ошибка при загрузке набора данных: {e}")
        raise

# Функция для разделения признаков и целевой переменной
def split_features_target(dataset, target_column):
    """
    Разделяет набор данных на признаки (X) и целевую переменную (y).
    """
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    return X, y

# Функция обучения модели методом итеративного обучения
def train_model_iterative(X, y, epochs=2000, learning_rate=0.05):
    """
    Обучает модель логистической регрессии с использованием SGDClassifier в итеративном режиме.
    """

    model = SGDClassifier(
        loss='log_loss',       # Логистическая регрессия
        penalty=None,          # Без регуляризации
        learning_rate='constant',
        eta0=learning_rate,
        max_iter=1,            # Обучение в 1 эпоху за вызов partial_fit
        warm_start=True,       # Для многократного вызова partial_fit
        random_state=0
    )

    # Преобразуем X и y в массивы numpy
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    errors = []
    accuracies = []

    # SGDClassifier требует список классов при первом вызове partial_fit
    classes = np.unique(y_np)

    for epoch in range(epochs):
        # partial_fit выполняет 1 эпоху обучения
        model.partial_fit(X_np, y_np, classes=classes)

        # Прогнозы вероятностей для расчета MSE
        pred_probs = model.predict_proba(X_np)[:, 1]
        mse = np.mean((pred_probs - y_np) ** 2)
        errors.append(mse)

        # Точность (accuracy)
        y_pred = model.predict(X_np)
        acc = accuracy_score(y_np, y_pred)
        accuracies.append(acc)

    return model, errors, accuracies

# Функция оценки модели
def evaluate_model(model, X, y):
    """
    Оценивает финальные метрики на всем наборе данных (в данном случае тренировочном).
    """
    y_pred = model.predict(X)
    metrics = {
        "Точность": accuracy_score(y, y_pred),
        "Точность (Precision)": precision_score(y, y_pred),
        "Полнота (Recall)": recall_score(y, y_pred),
        "F1-мера": f1_score(y, y_pred)
    }
    return metrics

# Функция сохранения модели
def save_model(model, file_path):
    """
    Сохраняет обученную модель в файл в формате pickle.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в файл: {file_path}")

# Функция построения графиков истории обучения
def plot_training_history(errors, accuracies, error_plot_path, accuracy_plot_path):
    """
    Сохраняет два графика:
      1) MSE от эпохи
      2) Точность (accuracy) от эпохи
    """
    # График MSE от эпохи
    plt.figure(figsize=(7, 5))
    plt.plot(range(len(errors)), errors, label='MSE')
    plt.title('Ошибка обучения (MSE) от эпохи')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(error_plot_path)
    plt.close()

    # График точности от эпохи
    plt.figure(figsize=(7, 5))
    plt.plot(range(len(accuracies)), accuracies, label='Точность', color='orange')
    plt.title('Точность обучения от эпохи')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.savefig(accuracy_plot_path)
    plt.close()

# Основной блок выполнения программы
if __name__ == "__main__":
    dataset_path = '../../datasets/normalized_shuffled_train.csv'
    model_save_path = '../model_lib_weights.pkl'

    error_plot_path = '../sklearn_error_plot.png'
    accuracy_plot_path = '../sklearn_accuracy_plot.png'

    target_column = 'TenYearCHD'
    EPOCHS = 2000
    LR = 0.05

    # Шаг 1: Загрузка набора данных
    dataset = load_dataset(dataset_path)

    # Шаг 2: Разделение на признаки и целевую переменную
    X_train, y_train = split_features_target(dataset, target_column)

    # Шаг 3: Обучение модели
    print("Начало обучения модели...")
    model, errors, accuracies = train_model_iterative(X_train, y_train, EPOCHS, LR)

    # Шаг 4: Построение графиков
    plot_training_history(errors, accuracies, error_plot_path, accuracy_plot_path)

    # Шаг 5: Оценка модели
    metrics = evaluate_model(model, X_train, y_train)
    print("Итоговые метрики обучения:", metrics)

    # Шаг 6: Сохранение модели
    save_model(model, model_save_path)
    print("Обучение завершено.")
